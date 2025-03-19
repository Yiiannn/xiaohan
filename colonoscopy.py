# 代码目的：基于streamlit包生成网页

# 导入包
import pandas as pd
import streamlit as st
import joblib
import pickle

# main function
# 设置网页名称
st.set_page_config(page_title='肠镜检查依从性评估工具')

# 设置网页标题
st.header('中老年人肠镜检查依从性评估网页工具\nWeb-based tool for colonoscopy compliance assessment in middle-aged and older adults')

# 设置副标题
st.subheader('欢迎使用本工具！\nWelcome to this tool! ')


# 在侧边栏添加说明
st.sidebar.info(
    "您可使用本工具预测肠镜检查的依从性。请注意，本预测结果仅供参考。\nYou can use this tool to predict compliance with colonoscopy. Please note that this forecast is for reference only.")

# 上传文件
uploaded_file = st.file_uploader("请上传包含特定信息数据的表格文件:\nPlease upload a form file containing specific information data:", type=["csv", "xlsx"])

# 判断用户是否上传了文件
if uploaded_file is not None:
    # 读取上传的文件
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file,encoding="gbk")
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file,encoding="gbk")

# 假设数据有三列，每列包含一个人的数据
    # 处理每个人的数据并预测
    def process_and_predict(data):
        # 预测每个人的结果
        results = []
        for index, row in data.iterrows():
            # 填充输入数据
            input_dict = {
                'gender': row['性别(Gender)'], 'a1': row['年龄(Age)'], 'a2': row['婚姻状况(Marital status)'], 'a3': row['受教育水平(Education)'],'a4': row['职业(Occupation)'], 'b1': row['慢性腹泻史(History of chronic diarrhea)'], 'b2': row['慢性便秘史(History of chronic constipation)'], 'b3': row['粘液和或血便史(History of mucus or blood in the stool)'], 'b4': row['慢性阑尾炎或阑尾切除史(History of chronic appendicitis or appendectomy)'], 'b5': row['慢性胆囊炎或胆囊切除史(History of chronic cholecystitis or cholecystectomy)'], 'b6': row['近10年来有无经历过对精神造成较大创伤或痛苦的事件(Psychological trauma or distress in the past 10 years)'], 'b7': row['癌症史(History of cancer)'],'b9': row['肠息肉史(History of colorectal polyps)'], 'b10': row['一级亲属(父、母、兄弟姐妹、子女)肠癌史(Family history of colorectal cancer in first-degree relatives)'], 'b11': row['血吸虫病史(History of schistosomiasis)'], 'riskassessment': row['危险度评估结果(Colorectal cancer risk assessment-based questionnaire)'],'c1': row['是否抽烟？(Smoking)'],'FOBT': row['便隐血测试(FOBT screening)']
            }

            input_df = pd.DataFrame([input_dict])

            # 对dataframe中传入的数据进行编码
            def coding_fun(input_df):
                input_df['gender'] = input_df['gender'].replace(['男性(Male)', '女性(Female)'], [1, 0])
                input_df['a2'] = input_df['a2'].replace(['已婚(Married)', '未婚(Unmarried/Divorced/Widowed)'], [1, 0])
                input_df['a3'] = input_df['a3'].replace(
                    ['小学及以下(primary school and below)', '中专、中学(middle school)', '大学、大专(university and above)'],
                    [1, 2, 3])
                input_df['a4'] = input_df['a4'].replace(['无业(Unemployed)', '企业(enterprises)', "事业单位(Government agency)", '农民(Farmer)', '其他(Others)'], [1, 2, 3, 4, 5])
                input_df['b1'] = input_df['b1'].replace(['否(No)', '是(Yes)'], [0,1])
                input_df['b2'] = input_df['b2'].replace(['否(No)', '是(Yes)'], [0,1])
                input_df['b3'] = input_df['b3'].replace(['否(No)', '是(Yes)'], [0,1])
                input_df['b4'] = input_df['b4'].replace(['否(No)', '是(Yes)'], [0,1])
                input_df['b5'] = input_df['b5'].replace(['否(No)', '是(Yes)'], [0,1])
                input_df['b6'] = input_df['b6'].replace(['否(No)', '是(Yes)'], [0,1])
                input_df['b7'] = input_df['b7'].replace(['否(No)', '是(Yes)'], [0,1])
                input_df['b9'] = input_df['b9'].replace(['否(No)', '是(Yes)'], [0,1])
                input_df['b10'] = input_df['b10'].replace(['否(No)', '是(Yes)','不清楚(Unknow)'], [0,1,2])
                input_df['b11'] = input_df['b11'].replace(['否(No)', '是(Yes)'], [0,1])
                input_df['riskassessment'] = input_df['riskassessment'].replace(['阴性(Negative)', '阳性(Positive)'], [0,1])
                input_df['c1'] = input_df['c1'].replace(['过去吸(Past cigarette smoking)', '现在吸(Current cigarette smoking)','不吸(Never cigarette smoking)'], [0,1,2])
                input_df['FOBT'] = input_df['FOBT'].replace(['阴性(Negative)', '阳性(Positive)'], [0,1])
                return input_df

            def make_predict(input_df):
                
                # Load the trained model for predict
                with open("E:/上中医/研0/论文写作/结直肠癌行为预测/实验更新_20250105/网页工具/sklearn_GBM_best_model.sav", "rb") as f:
                    model = pickle.load(f)
                #model = joblib.load("E:/上中医/研0/论文写作/结直肠癌行为预测/实验更新_20250105/机器学习模型/1_机器学习模型训练/Model_Parameters/XGBoost_best_model.sav")
                # make prediction
                predict_result = model.predict(input_df)  # 对输入的数据进行预测

                # check probability
                predict_probability = model.predict_proba(input_df)  # 给出预测概率
                return predict_result, predict_probability

            input_df1 = coding_fun(input_df=input_df)
            result, probability = make_predict(input_df=input_df1)

            # 结果处理
            if int(result) == 1:
                results.append("属于低依从性人群，需对其进行个性化干预\nIndividuals with low adherence require personalized interventions")
            else:
                results.append("属于高依从性人群，无需对其进行干预\nIndividuals with high adherence do not require intervention")
        return results

    # 执行并显示结果
    results = process_and_predict(data)
    
    # 显示多个人的风险评估结果
    st.header("肠镜检查依从性评估结果：\nResults of colonoscopy adherence assessment:")
    for i, result in enumerate(results, 1):
        st.write(f"第{i}个人: {result}")

else:
    st.warning("请上传一个包含特定信息数据的表格文件！\nPlease upload a spreadsheet file containing specific information data.")

