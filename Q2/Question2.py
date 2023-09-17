import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import scipy.stats as stats

def plot_acf_pacf(ts):
    #绘制自相关系数和偏自相关系数
    plot_acf(ts)
    plt.xlabel('Hysteresis number')
    plt.ylabel('Autocorrelation coefficient')
    plt.grid(True)
    plt.show()

    plot_pacf(ts)
    plt.xlabel('Hysteresis number')
    plt.ylabel('Partial autocorrelation coefficient')
    plt.grid(True)
    plt.show()

def Actual_Predict_plt(actual,predict):
    #绘制预测结果
    plt.plot(actual, label='Actual')
    plt.plot(predict, label='Predicted')
    plt.grid(True)
    plt.legend()
    plt.show()

def Q_Q_plt(data):
    stats.probplot(data, dist="norm", plot=plt)
    plt.show()


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))





#18号
def M_U():
    # 读取数据
    data = pd.read_excel('M_U.xlsx')
    # 提取“发货-收货”站点城市对及其对应的快递运输数量，并将其转换为时间序列格式
    ts = pd.Series(data['快递运输数量(件) (Express delivery quantity (PCS))'].values,
                   index=pd.to_datetime(data['日期(年/月/日) (Date Y/M/D)'].values))

    # 以天为单位重采样，并对缺失值进行插值处理
    ts_day = ts.resample('D').sum().interpolate()

    # #划分数据集和测试集
    train_data = ts_day
    test_data = ts_day

    # 绘制自相关系数和偏自相关系数图像
    plot_acf_pacf(train_data)
    ts_day_diff = train_data.diff().dropna()
    plot_acf_pacf(ts_day_diff)

    # ADF
    result = adfuller(ts_day_diff)
    print('The result of ADF: ')
    print(result)

    # 输出预处理后的数据
    print(ts_day.head())

    # 选择p、d、q值
    p, d, q = 1, 1, 1

    # 拟合ARIMA模型
    model = ARIMA(train_data, order=(p, d, q)).fit()

    # 预测数据
    pred_0410_0417 = model.predict(start='2018-04-19', end='2019-04-17', dynamic=False)
    # pred_0410_0417 = model.forecast(steps=49)
    Actual_Predict_plt(test_data, pred_0410_0417)

    # 绘制Q_Q图
    Q_Q_plt(ts)

    # 采用 MAE 评价模型
    MAE = mean_absolute_error(test_data, pred_0410_0417)
    print("MAE = ", MAE)


    # 预测2019年4月18日和2019年4月19日各“发货-收货”站点城市之间快递运输数量
    pred = model.predict(start='2019-04-18', end='2019-04-19')

    # 输出预测结果
    print(pred)
def Q_V():
    # 读取数据
    data = pd.read_excel('Q_V.xlsx')
    # 提取“发货-收货”站点城市对及其对应的快递运输数量，并将其转换为时间序列格式
    ts = pd.Series(data['快递运输数量(件) (Express delivery quantity (PCS))'].values,
                   index=pd.to_datetime(data['日期(年/月/日) (Date Y/M/D)'].values))

    # 以天为单位重采样，并对缺失值进行插值处理
    ts_day = ts.resample('D').sum().interpolate()

    # #划分数据集和测试集
    train_data = ts_day
    test_data = ts_day

    # 绘制自相关系数和偏自相关系数图像
    plot_acf_pacf(train_data)
    ts_day_diff = train_data.diff().dropna()
    plot_acf_pacf(ts_day_diff)

    # ADF
    result = adfuller(ts_day_diff)
    print('The result of ADF: ')
    print(result)

    # 输出预处理后的数据
    print(ts_day.head())

    # 选择p、d、q值
    p, d, q = 0, 1, 0

    # 拟合ARIMA模型
    model = ARIMA(train_data, order=(p, d, q)).fit()

    # 预测数据
    pred_0410_0417 = model.predict(start='2018-04-19', end='2019-04-17', dynamic=False)
    # pred_0410_0417 = model.forecast(steps=49)
    Actual_Predict_plt(test_data, pred_0410_0417)

    # 绘制Q_Q图
    Q_Q_plt(ts_day)

    # 采用 MAE，MSE 评价模型
    MAE = mean_absolute_error(test_data, pred_0410_0417)
    print("MAE = ", MAE)
    # MSE = mse(test_data,pred_0410_0417)
    # print("MSE = ",MSE)

    # 预测2019年4月18日和2019年4月19日各“发货-收货”站点城市之间快递运输数量
    pred = model.predict(start='2019-04-18', end='2019-04-19')

    # 输出预测结果
    print(pred)
def K_L():
    # 读取数据
    data = pd.read_excel('K_L.xlsx')
    # 提取“发货-收货”站点城市对及其对应的快递运输数量，并将其转换为时间序列格式
    ts = pd.Series(data['快递运输数量(件) (Express delivery quantity (PCS))'].values,
                   index=pd.to_datetime(data['日期(年/月/日) (Date Y/M/D)'].values))

    # 以天为单位重采样，并对缺失值进行插值处理
    ts_day = ts.resample('D').sum().interpolate()

    # #划分数据集和测试集
    train_data = ts_day
    test_data = ts_day

    # 绘制自相关系数和偏自相关系数图像
    plot_acf_pacf(train_data)
    # ts_day_diff = train_data.diff().dropna()
    # plot_acf_pacf(ts_day_diff)

    # ADF
    result = adfuller(train_data)
    print('The result of ADF: ')
    print(result)

    # 输出预处理后的数据
    print(ts_day.head())

    # 选择p、d、q值
    p, d, q =2, 0, 1

    # 拟合ARIMA模型
    model = ARIMA(train_data, order=(p, d, q)).fit()

    # 预测数据
    pred_0410_0417 = model.predict(start='2018-04-19', end='2019-04-17', dynamic=False)
    # pred_0410_0417 = model.forecast(steps=49)
    Actual_Predict_plt(test_data, pred_0410_0417)

    # 绘制Q_Q图
    Q_Q_plt(ts)

    # 采用 MAE，MSE 评价模型
    MAE = mean_absolute_error(test_data, pred_0410_0417)
    print("MAE = ", MAE)
    # MSE = mse(test_data,pred_0410_0417)
    # print("MSE = ",MSE)

    # 预测2019年4月18日和2019年4月19日各“发货-收货”站点城市之间快递运输数量
    pred = model.predict(start='2019-04-18', end='2019-04-19')

    # 输出预测结果
    print(pred)
def G_V():
    # 读取数据
    data = pd.read_excel('G_V.xlsx')
    # 提取“发货-收货”站点城市对及其对应的快递运输数量，并将其转换为时间序列格式
    ts = pd.Series(data['快递运输数量(件) (Express delivery quantity (PCS))'].values,
                   index=pd.to_datetime(data['日期(年/月/日) (Date Y/M/D)'].values))

    # 以天为单位重采样，并对缺失值进行插值处理
    ts_day = ts.resample('D').sum().interpolate()

    # #划分数据集和测试集
    train_data = ts_day
    test_data = ts_day

    # 绘制自相关系数和偏自相关系数图像
    plot_acf_pacf(train_data)
    # ts_day_diff = train_data.diff().dropna()
    # plot_acf_pacf(ts_day_diff)

    # ADF
    result = adfuller(train_data)
    print('The result of ADF: ')
    print(result)

    # 输出预处理后的数据
    print(ts_day.head())

    # 选择p、d、q值
    p, d, q = 5, 0, 9

    # 拟合ARIMA模型
    model = ARIMA(train_data, order=(p, d, q)).fit()

    # 预测数据
    pred_0410_0417 = model.predict(start='2018-04-19', end='2019-04-17', dynamic=False)
    # pred_0410_0417 = model.forecast(steps=49)
    Actual_Predict_plt(test_data, pred_0410_0417)

    # 绘制Q_Q图
    Q_Q_plt(ts)

    # 采用 MAE，MSE 评价模型
    MAE = mean_absolute_error(test_data, pred_0410_0417)
    print("MAE = ", MAE)
    # MSE = mse(test_data,pred_0410_0417)
    # print("MSE = ",MSE)

    # 预测2019年4月18日和2019年4月19日各“发货-收货”站点城市之间快递运输数量
    pred = model.predict(start='2019-04-18', end='2019-04-19')

    # 输出预测结果
    print(pred)

#19号
def V_G():
    # 读取数据
    data = pd.read_excel('V_G.xlsx')
    # 提取“发货-收货”站点城市对及其对应的快递运输数量，并将其转换为时间序列格式
    ts = pd.Series(data['快递运输数量(件) (Express delivery quantity (PCS))'].values,
                   index=pd.to_datetime(data['日期(年/月/日) (Date Y/M/D)'].values))

    # 以天为单位重采样，并对缺失值进行插值处理
    ts_day = ts.resample('D').sum().interpolate()

    # #划分数据集和测试集
    train_data = ts_day
    test_data = ts_day

    # 绘制自相关系数和偏自相关系数图像
    plot_acf_pacf(train_data)
    # ts_day_diff = train_data.diff().dropna()
    # plot_acf_pacf(ts_day_diff)

    # ADF
    result = adfuller(train_data)
    print('The result of ADF: ')
    print(result)

    # 输出预处理后的数据
    print(ts_day.head())

    # 选择p、d、q值
    p, d, q =4, 0, 7

    # 拟合ARIMA模型
    model = ARIMA(train_data, order=(p, d, q)).fit()

    # 预测数据
    pred_0410_0417 = model.predict(start='2018-04-19', end='2019-04-17', dynamic=False)
    # pred_0410_0417 = model.forecast(steps=49)
    Actual_Predict_plt(test_data, pred_0410_0417)

    # 绘制Q_Q图
    Q_Q_plt(ts)

    # 采用 MAE，MSE 评价模型
    MAE = mean_absolute_error(test_data, pred_0410_0417)
    print("MAE = ", MAE)
    # MSE = mse(test_data,pred_0410_0417)
    # print("MSE = ",MSE)

    # 预测2019年4月18日和2019年4月19日各“发货-收货”站点城市之间快递运输数量
    pred = model.predict(start='2019-04-18', end='2019-04-19')

    # 输出预测结果
    print(pred)
def A_Q():
    # 读取数据
    data = pd.read_excel('A_Q.xlsx')
    # 提取“发货-收货”站点城市对及其对应的快递运输数量，并将其转换为时间序列格式
    ts = pd.Series(data['快递运输数量(件) (Express delivery quantity (PCS))'].values,
                   index=pd.to_datetime(data['日期(年/月/日) (Date Y/M/D)'].values))

    # 以天为单位重采样，并对缺失值进行插值处理
    ts_day = ts.resample('D').sum().interpolate()

    # #划分数据集和测试集
    train_data = ts_day
    test_data = ts_day

    # 绘制自相关系数和偏自相关系数图像
    plot_acf_pacf(train_data)
    # ts_day_diff = train_data.diff().dropna()
    # plot_acf_pacf(ts_day_diff)

    # ADF
    result = adfuller(train_data)
    print('The result of ADF: ')
    print(result)

    # 输出预处理后的数据
    print(ts_day.head())

    # 选择p、d、q值
    p, d, q = 4, 0, 11

    # 拟合ARIMA模型
    model = ARIMA(train_data, order=(p, d, q)).fit()

    # 预测数据
    pred_0410_0417 = model.predict(start='2018-04-19', end='2019-04-17', dynamic=False)
    # pred_0410_0417 = model.forecast(steps=49)
    Actual_Predict_plt(test_data, pred_0410_0417)

    # 绘制Q_Q图
    Q_Q_plt(ts)

    # 采用 MAE，MSE 评价模型
    MAE = mean_absolute_error(test_data, pred_0410_0417)
    print("MAE = ", MAE)
    # MSE = mse(test_data,pred_0410_0417)
    # print("MSE = ",MSE)

    # 预测2019年4月18日和2019年4月19日各“发货-收货”站点城市之间快递运输数量
    pred = model.predict(start='2019-04-18', end='2019-04-19')

    # 输出预测结果
    print(pred)
def D_A():
    # 读取数据
    data = pd.read_excel('D_A.xlsx')
    # 提取“发货-收货”站点城市对及其对应的快递运输数量，并将其转换为时间序列格式
    ts = pd.Series(data['快递运输数量(件) (Express delivery quantity (PCS))'].values,
                   index=pd.to_datetime(data['日期(年/月/日) (Date Y/M/D)'].values))

    # 以天为单位重采样，并对缺失值进行插值处理
    ts_day = ts.resample('D').sum().interpolate()

    # #划分数据集和测试集
    train_data = ts_day
    test_data = ts_day

    # 绘制自相关系数和偏自相关系数图像
    plot_acf_pacf(train_data)
    ts_day_diff = train_data.diff().dropna()
    plot_acf_pacf(ts_day_diff)

    # ADF
    result = adfuller(ts_day_diff)
    print('The result of ADF: ')
    print(result)

    # 输出预处理后的数据
    print(ts_day.head())

    # 选择p、d、q值
    p, d, q = 2, 1, 2

    # 拟合ARIMA模型
    model = ARIMA(train_data, order=(p, d, q)).fit()

    # 预测数据
    pred_0410_0417 = model.predict(start='2018-04-19', end='2019-04-17', dynamic=False)
    # pred_0410_0417 = model.forecast(steps=49)
    Actual_Predict_plt(test_data, pred_0410_0417)

    # 绘制Q_Q图
    Q_Q_plt(ts)

    # 采用 MAE，MSE 评价模型
    MAE = mean_absolute_error(test_data, pred_0410_0417)
    print("MAE = ", MAE)
    # MSE = mse(test_data,pred_0410_0417)
    # print("MSE = ",MSE)

    # 预测2019年4月18日和2019年4月19日各“发货-收货”站点城市之间快递运输数量
    pred = model.predict(start='2019-04-18', end='2019-04-19')

    # 输出预测结果
    print(pred)
def L_K():
    # 读取数据
    data = pd.read_excel('L_K.xlsx')
    # 提取“发货-收货”站点城市对及其对应的快递运输数量，并将其转换为时间序列格式
    ts = pd.Series(data['快递运输数量(件) (Express delivery quantity (PCS))'].values,
                   index=pd.to_datetime(data['日期(年/月/日) (Date Y/M/D)'].values))

    # 以天为单位重采样，并对缺失值进行插值处理
    ts_day = ts.resample('D').sum().interpolate()

    # #划分数据集和测试集
    train_data = ts_day
    test_data = ts_day

    # 绘制自相关系数和偏自相关系数图像
    plot_acf_pacf(train_data)
    # ts_day_diff = train_data.diff().dropna()
    # plot_acf_pacf(ts_day_diff)

    # ADF
    result = adfuller(train_data)
    print('The result of ADF: ')
    print(result)

    # 输出预处理后的数据
    print(ts_day.head())

    # 选择p、d、q值
    p, d, q =1,0,0

    # 拟合ARIMA模型
    model = ARIMA(train_data, order=(p, d, q)).fit()

    # 预测数据
    pred_0410_0417 = model.predict(start='2018-04-19', end='2019-04-17', dynamic=False)
    # pred_0410_0417 = model.forecast(steps=49)
    Actual_Predict_plt(test_data, pred_0410_0417)


    # 采用 MAE，MSE 评价模型
    MAE = mean_absolute_error(test_data, pred_0410_0417)
    print("MAE = ", MAE)


    # 预测2019年4月18日和2019年4月19日各“发货-收货”站点城市之间快递运输数量
    pred = model.predict(start='2019-04-18', end='2019-04-19')

    # 输出预测结果
    print(pred)

#18号和19号总的
def All_data():
    # 读取数据
    data = pd.read_excel('B题-附件1.xlsx')
    # 提取“发货-收货”站点城市对及其对应的快递运输数量，并将其转换为时间序列格式
    ts = pd.Series(data['快递运输数量(件) (Express delivery quantity (PCS))'].values,
                   index=pd.to_datetime(data['日期(年/月/日) (Date Y/M/D)'].values))

    # 以天为单位重采样，并对缺失值进行插值处理
    ts_day = ts.resample('D').sum().interpolate()

    # #划分数据集和测试集
    train_data = ts_day
    test_data = ts_day

    # 绘制自相关系数和偏自相关系数图像
    plot_acf_pacf(train_data)
    # ts_day_diff = train_data.diff().dropna()
    # plot_acf_pacf(ts_day_diff)

    # ADF
    result = adfuller(train_data)
    print('The result of ADF: ')
    print(result)

    # 输出预处理后的数据
    print(ts_day.head())

    # 选择p、d、q值
    p, d, q = 3, 0, 0

    # 拟合ARIMA模型
    model = ARIMA(train_data, order=(p, d, q)).fit()

    # 预测数据
    pred_0410_0417 = model.predict(start='2018-04-19', end='2019-04-17', dynamic=False)
    # pred_0410_0417 = model.forecast(steps=49)
    Actual_Predict_plt(test_data, pred_0410_0417)

    # 绘制Q_Q图
    Q_Q_plt(ts)

    # 采用 MAE，MSE 评价模型
    MAE = mean_absolute_error(test_data, pred_0410_0417)
    print("MAE = ", MAE)
    # MSE = mse(test_data,pred_0410_0417)
    # print("MSE = ",MSE)

    # 预测2019年4月18日和2019年4月19日各“发货-收货”站点城市之间快递运输数量
    pred = model.predict(start='2019-04-18', end='2019-04-19')

    # 输出预测结果
    print(pred)


if __name__ == '__main__':
    #  18
    # M_U()
    # Q_V()
    # K_L()
    # G_V()

    # 19
    V_G()
    # A_Q()
    # D_A()
    # L_K()
    #
    # #all
    # All_data()