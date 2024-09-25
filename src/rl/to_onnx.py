import sys
from PPO import PPO
import torch
import os
ppo = PPO(None, None, None, False)
load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model28-08-2024_10-19-49"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model28-08-2024_10-19-49"
num = 760
ppo.load_model(load_actor_model_path, load_critic_model_path, num)

dir = os.path.dirname(os.path.realpath(__file__)) + "/px4_model/28-08-2024_10-19-49"
if not (os.path.exists(dir)):
    os.makedirs(dir)
input_tensor = torch.rand((1, ppo.env.state_dim), dtype=torch.float32)
torch.onnx.export(
    ppo.actor.mean_net,                  # model to export
    (input_tensor,),        # inputs of the model,
    dir+"/model.onnx",        # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    verbose=False             # True or False to select the exporter to use
)
# print(ppo.actor.mean_net[0].weight)

# from PPO import PPO
# import torch
# import os
# if __name__=="__main__":
#     ppo = PPO(None, None, None, False)
#     load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model28-08-2024_10-19-49"
#     load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model28-08-2024_10-19-49"
#     num = 760
#     ppo.load_model(load_actor_model_path, load_critic_model_path, 360)

#     dir = os.path.dirname(os.path.realpath(__file__)) + "/deepCmodel/28-08-2024_10-19-49"
#     if not (os.path.exists(dir)):
#         os.makedirs(dir)
#     torch.save(ppo.actor.mean_net, dir+"/mean_net.pt")

# input_tensor = torch.rand((1, ppo.env.state_dim), dtype=torch.float32)
# torch.onnx.export(
#     ppo.actor.mean_net,                  # model to export
#     (input_tensor,),        # inputs of the model,
#     dir+"/model.onnx",        # filename of the ONNX model
#     input_names=["input"],  # Rename inputs for the ONNX model
#     verbose=True             # True or False to select the exporter to use
# )
# print(ppo.actor.mean_net[0].weight)

quantized_model = torch.quantization.quantize_dynamic(ppo.actor.mean_net, {torch.nn.Linear,torch.nn.Tanh}, dtype=torch.qint8)
scripted_model = torch.jit.script(quantized_model)
torch.jit.save(scripted_model, dir+"/mobile.pt")

# # 获取模型的 state_dict，它包含所有的参数
state_dict = quantized_model.state_dict()
print(state_dict["4.scale"].numpy())
# # 打印所有的参数名称和值
# for name, param in state_dict.items():
#     print(f"Parameter name: {name}")
#     print(f"Parameter value: {param.detach().numpy()}\n")


# # 创建文件保存所有权重和偏置
with open(dir+'/quantized_model_weights.c', 'w') as f:
    # 写入量化推理函数的C代码
    f.write("""
#define Relu(x) x<0.f?0.f:x            

// 量化推理函数
void quantized_inference(
    const float* input, const int8_t* weight, const float* bias,
    float scale_w, int weight_rows, int weight_cols, float* output)
{
    // 中间结果累积使用 int32_t
    int32_t y_int32[weight_rows];

    // 初始化输出为 0
    for (int i = 0; i < weight_rows; i++) {
        y_int32[i] = 0;
    }

    // 矩阵乘法：计算量化的矩阵乘法结果
    for (int i = 0; i < weight_rows; i++) {
        for (int j = 0; j < weight_cols; j++) {
            // 累积乘法：输出 = 量化权重 * 量化输入
            y_int32[i] += weight[i * weight_cols + j] * input[j];
        }
    }

    // 反量化：将 int32 转换为浮点数，并加上偏置
    for (int i = 0; i < weight_rows; i++) {
        output[i] = scale_w * y_int32[i] + bias[i];
    }
}


""")

    # 初始化层编号
    layer_num = 1
    
    # 遍历 state_dict 中的每个键值对
    for name, param in state_dict.items():
        if "_packed_params._packed_params" in name:
            packed_params = param[0]  # 获取量化权重
            bias = param[1]  # 获取偏置
            
            # 提取量化后的 int8 权重数据
            weights = packed_params.int_repr().numpy()  # 获取原始 int8 数据
            scale = packed_params.q_scale()  # 获取 scale
            zero_point = packed_params.q_zero_point()  # 获取 zero_point
            
            bias_data = bias.detach().numpy()

            shape = weights.shape

            # 保存权重为二维数组 (int8)
            f.write(f"// Layer {layer_num} weights and bias\n")
            f.write(f"const int8_t w{layer_num}[{shape[0]}][{shape[1]}] = {{\n")
            for row in weights:
                f.write("    {" + ", ".join(map(str, row)) + "},\n")
            f.write("};\n\n")

            # 保存量化信息
            f.write(f"const float scale{layer_num} = {scale};\n")
            f.write(f"const int zero_point{layer_num} = {zero_point};\n\n")

            # 保存偏置为一维数组
            f.write(f"const float b{layer_num}[{bias_data.shape[0]}] = {{")
            f.write(", ".join(map(str, bias_data)))
            f.write("};\n\n")

            # 写入示例代码，说明如何调用推理函数
            f.write(f"""// 使用第 {layer_num} 层权重和偏置的推理函数示例
void inference_layer_{layer_num}(const int8_t* input, float* output) {{
    quantized_inference(
        input, &w{layer_num}[0][0], b{layer_num},
        scale{layer_num}, {shape[0]}, {shape[1]}, output);
        for(int i = 0; i < {shape[0]}; i++)
            output[i] = Relu(output[i]);
}}

""")
            layer_num += 1


print("量化模型的权重和偏置已保存到 quantized_model_weights.c 文件中！")