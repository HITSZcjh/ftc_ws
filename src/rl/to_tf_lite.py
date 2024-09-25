import onnx
from onnx_tf.backend import prepare
import os
import tensorflow as tf
from tensorflow.lite.python.util import convert_bytes_to_c_source

if __name__=="__main__":
    dir = os.path.dirname(os.path.realpath(__file__)) + "/px4_model/28-08-2024_10-19-49"

    onnx_model = onnx.load(dir+"/model.onnx")  # load onnx model
    output = prepare(onnx_model)  # run the loaded model
    output.export_graph(dir+"/model.tf")  # export the model
    # saved_model = tf.saved_model.load(dir+"/model.tf")
    converter = tf.lite.TFLiteConverter.from_saved_model(dir+'/model.tf') # path to the SavedModel directory
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    # # Save the model.
    with open(dir+'/model.tflite', 'wb') as f:
      f.write(tflite_model)


    # # 加载 tflite 模型
    interpreter = tf.lite.Interpreter(model_path=dir+'/model.tflite')
    interpreter.allocate_tensors()

    # 获取模型的张量信息
    tensor_details = interpreter.get_tensor_details()

    # 保存文件的路径
    output_file = dir+'/weights_output.txt'
    # 打开一个文件，准备将所有层的权重和偏置写入
    with open(output_file, 'w') as f:
        # 遍历所有张量
        for i, layer in enumerate(interpreter.get_tensor_details()):
            print(i, layer)
            if 'weights' in layer['name'] or 'bias' in layer['name']:
                # 提取权重或偏置
                tensor = interpreter.tensor(layer['index'])()
                shape = tensor.shape

                # 保存权重为二维数组，偏置为一维数组
                if len(shape) == 2:
                    # 这是权重，假设为二维数组
                    weights = tensor
                    f.write(f"const float w{i+1}[{shape[0]}][{shape[1]}]={{")
                    for row in weights:
                        f.write("{" + ",".join(map(str, row)) + "},")
                    f.write("};\n")

                elif len(shape) == 1:
                    # 这是偏置，假设为一维数组
                    biases = tensor
                    f.write(f"const float b{i+1}[{shape[0]}]={{")
                    f.write(",".join(map(str, biases)))
                    f.write("};\n")

    print("所有层的权重和偏置已保存到 model_weights_biases.txt 文件中！")


