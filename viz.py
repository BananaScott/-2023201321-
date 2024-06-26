import pandas as pd
import matplotlib.pyplot as plt
df_false = pd.read_csv('result_train_False.csv')  # 从.csv文件中读取训练结果并存储在dataframe中
df_true = pd.read_csv('result_train_True.csv')


plt.figure(dpi=100)  # 创建一个图像并将dpi设置为300
plt.plot(df_false['test loss'])
plt.plot(df_true['test loss']) # 绘制用真实数据训练的模型的测试损失和生成数据训练的模型的测试损失
plt.plot(df_false['test acc'])
plt.plot(df_true['test acc'])  # 绘制使用真实数据训练的模型的测试准确性和生成数据训练的模型的测试准确性
plt.legend(['Test Loss', 'Test Loss with Gen Data', 'Test Acc', 'Test Acc with Gen Data'])  # 添加Legend标签，并指定它所代表的标签
plt.grid()  # 绘制网格以方便查看
plt.xlabel('Epochs')
plt.ylabel('Loss Value / Test Accuracy')  # 绘制X轴和Y轴的标签
plt.show()  # 将图像进行展示

