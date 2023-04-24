# 存储实验结果
pred_label = [1, 2, 3, 4]
pred_label = [str(item) for item in pred_label]
open('./res/ChengDu/data2_res.txt', mode='a').writelines(' '.join(pred_label) + '\n')
