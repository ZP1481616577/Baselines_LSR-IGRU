## 1_generate_data.py 
    生成原始的数据文件(data_2018_2022.pkl)和标签文件(data_2018_2022_label.pkl)，可以跳过这一步

## 2_process_data.py
    1中的两个文件预处理，得到模型所需要的数据文件(process_2018_2022_data_label.pkl)
    数据在链接: https://pan.baidu.com/s/1_ma6599-W5OtZXS1JydnHA 提取码: d0dq，7天有效尽快下载


## 3_generate_relation.py
    利用2中的文件，生成每个月中股票的关系图在relation文件夹下
    也可以生成每天的关系图，需要自己修改代码

## 4_generate_data.py
    利用2中的文件以及relation下的文件，生成模型所需要的每天的数据在data_train_predict文件夹下和kdcode文件夹下

## 5_model_train_predict.py
    利用4生成的数据训练模型，模型保存在model_saved文件夹下，
    训练完成后预测结果保存在prediction文件夹下