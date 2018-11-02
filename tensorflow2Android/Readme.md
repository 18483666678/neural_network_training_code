# Readme
Add date: 20181102

## File list
tsf2android.py

## Key points
1. 给输入数据命名为input,在android端需要用这个input来为输入数据赋值
2. 给输输数据命名为output,在android端需要用这个output来为获取输出的值
3. 不能使用 tf.train.write_graph()保存模型，因为它只是保存了模型的结构，并不保存训练完毕的参数值
4. 不能使用 tf.train.saver()保存模型，因为它只是保存了网络中的参数值，并不保存模型的结构。
5. graph_util.convert_variables_to_constants可以把整个sesion当作常量都保存下来，通过output_node_names参数来指定输出
6. tf.gfile.FastGFile('model/cxq.pb', mode='wb')指定保存文件的路径以及读写方式
7. f.write（output_graph_def.SerializeToString()）将固化的模型写入到文件

## Reference
[将tensorflow训练好的模型移植到android](https://www.jianshu.com/p/ddeb0400452f)