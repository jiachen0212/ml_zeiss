# coding=utf-8
'''
批量上传场景 算法demo

'''

from typing import List
import numpy as np
import json
import random
import time

from numpy.lib.shape_base import tile


class ZeissModel:
    def __init__(self, model_path: str):
        self._model = model_path

    def run(self, data_dir: str) -> str:
        '''
        data_dir: 输入数据文件夹

        double_cx_cc_LabCurve_list: [double_lab_curve, cx_lab_curve, cc_lab_curve] 
        每个元素均是81维的浮点值list

        deta_cx_thickness: [0.1, 0.1, 0.1, 0.1, 0.1.5]  正面的n个推荐膜厚浮点修改量
        deta_cc_thickness: [0.1, 0.1, 0.1, 0.1, 0.1.5]  背面的n个推荐膜厚浮点修改量

        modified_cx_thickness: [0.1, 0.1, 0.1, 0.1, 0.1.5]  正面的n个推荐膜厚 
        modified_cc_thickness: [0.1, 0.1, 0.1, 0.1, 0.1.5]  背面的n个推荐膜厚 
        
        original_lab_cure_similarity: 0.85
        modified_lab_cure_similarity: 0.95 

        modified_LAB: [4.81, 2.25, -13.74]
        
        cx_base_thickness: [25.0, 5, 13.0, 37.0, 102.0, 91.5, 35]
        cc_base_thickness: [25.0, 5, 13.0, 37.0, 102.0, 91.5, 35]
        thick_material: ['SiO2', 'ITO', 'Ti3O5', 'SiO2', 'Ti3O5','SiO2', 'COTEC1200']

        all_res = dict()  # 炉号为key, 获取每一炉的结果.

        '''

        # time.sleep(5)

        response_data: List = []
        error_msg :str = ""
        model_verion :str = "v0.0.1"

        oven_numbers:List[str] = ['33321091601','33321091602']

        for num in oven_numbers:
            res =  self.predict(num)
            response_data.append(res)

        response = {
            "status": "OK",
            "error_msg": error_msg,
            "model_version":model_verion,
            "data":response_data
        }
        
        data = json.dumps(response)
        # with open('./data/zeiss_demo_1013.json', 'w') as js_file:
        with open('./zeiss_demo_1013.json', 'w') as js_file:
            js_file.write(data)

        return json.dumps(response)

    def predict(self, oven_num: str) -> dict:

        double_lab_curve: List[float] = [6.07, 4.1, 2.47, 1.14, 0.9, 0.49, 0.51, 0.63, 1.04, 1.13, 1.46, 1.74, 1.99, 2.11, 2.08, 2.06, 2.04, 2.02, 1.98, 1.84, 1.67, 1.55, 1.47, 1.41, 1.31, 1.18, 1.12, 0.96, 0.87, 0.8, 0.74, 0.66, 0.58, 0.51, 0.5, 0.45, 0.38, 0.36,
                                         0.3, 0.29, 0.23, 0.24, 0.16, 0.12, 0.06, 0.04, 0.07, 0.1, 0.14, 0.06, 0.01, 0.03, 0.09, 0.08, 0.16, 0.23, 0.38, 0.54, 0.64, 0.75, 0.87, 1.02, 1.09, 1.25, 1.44, 1.62, 1.86, 2.09, 2.39, 2.63, 2.87, 3.13, 3.36, 3.68, 4.14, 4.27, 4.64, 4.76, 5.19, 5.51, 5.88]
        cx_lab_curve: List[float] = [6.6524, 4.5071, 2.2922, 1.1451, 0.7214, 0.4108, 0.2415, 0.3834, 0.5492, 0.5964, 0.6647, 0.8838, 1.0936, 1.0963, 0.992, 0.972, 1.0551, 1.1061, 1.0686, 0.927, 0.8003, 0.7419, 0.7491, 0.7563, 0.7154, 0.6076, 0.4921, 0.4414, 0.4152, 0.4251, 0.4191, 0.3947, 0.3412, 0.2696, 0.2235, 0.1996, 0.1907, 0.1965, 0.1844,
                                     0.1592, 0.1281, 0.0889, 0.0598, 0.0446, 0.031, 0.0375, 0.0398, 0.0386, 0.0287, 0.0257, 0.0249, 0.0326, 0.0593, 0.093, 0.1355, 0.1803, 0.2255, 0.268, 0.3089, 0.3568, 0.4025, 0.4695, 0.5404, 0.6328, 0.7406, 0.8652, 1.0015, 1.1388, 1.2783, 1.4115, 1.543, 1.6709, 1.8066, 1.9292, 2.0527, 2.1884, 2.3387, 2.5097, 2.6949, 2.8866, 3.1036]
        cc_lab_curve: List[float] = [7.3356, 4.8663, 2.5198, 1.4276, 0.8527, 0.4359, 0.3289, 0.4565, 0.5573, 0.6074, 0.8496, 1.0507, 1.1332, 1.0802, 1.0985, 1.202, 1.2691, 1.2089, 1.065, 0.9647, 0.9341, 0.9569, 0.9191, 0.8223, 0.6908, 0.6092, 0.5654, 0.5678, 0.5585, 0.5084, 0.4354, 0.3652, 0.3153, 0.301, 0.2963, 0.2923, 0.2769, 0.2354, 0.1909,
                                     0.158, 0.1339, 0.1295, 0.1184, 0.1111, 0.0976, 0.0863, 0.0688, 0.064, 0.0584, 0.0639, 0.0748, 0.0901, 0.1025, 0.1135, 0.1261, 0.1485, 0.1761, 0.2271, 0.2806, 0.3443, 0.4229, 0.5029, 0.5719, 0.6471, 0.7214, 0.788, 0.8605, 0.9497, 1.052, 1.1625, 1.2945, 1.4421, 1.6149, 1.7815, 1.947, 2.1099, 2.2678, 2.4232, 2.5676, 2.7107, 2.8521]

        for i, item in enumerate(double_lab_curve):
            double_lab_curve[i] = round(item + random.uniform(-1, 1.0), 5)

        for i, item in enumerate(cx_lab_curve):
            cx_lab_curve[i] = round(item + random.uniform(-1, 1.0), 5)

        for i, item in enumerate(cc_lab_curve):
            cc_lab_curve[i] = round(item + random.uniform(-1, 1.0), 5)

        # 2,7层膜厚不变
        cx_deta_thickness: List[float] = [random.uniform(-1, 1), 0, random.uniform(
            -1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), 0]

        cc_deta_thickness: List[float] = [random.uniform(-1, 1), 0, random.uniform(
            -1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), 0]


        # 正背base 膜厚
        cx_base_thickness: List[float] = [25.0, 5, 13.0, 37.0, 102.0, 91.5, 35]
        cc_base_thickness: List[float] = [25.0, 5, 13.0, 37.0, 102.0, 91.5, 35]

        # modified thickness
        cx_modified_thickness: List[float] = [
            cx_base_thickness[i] + cx_deta_thickness[i] for i in range(len(cx_deta_thickness))]

        cc_modified_thickness: List[float] = [
        cc_base_thickness[i] + cc_deta_thickness[i] for i in range(len(cc_deta_thickness))]

        # 镜片各层镀膜材质 可从cxcc正背的evt中获取
        cx_thick_material: List[str] = ['SiO2', 'ITO',
                                     'Ti3O5', 'SiO2', 'Ti3O5', 'SiO2', 'COTEC1200']

        cc_thick_material: List[str] = ['SiO2', 'ITO',
                                     'Ti3O5', 'SiO2', 'Ti3O5', 'SiO2', 'COTEC1200']

        original_lab_cure_similarity: float = random.uniform(0.5, 1.0)
        modified_lab_cure_similarity: float = random.uniform(0.3, 1.0)
        modified_LAB: List[float] = [random.uniform(
            3.3, 6.8), random.uniform(-2.0, 2.0), random.uniform(-15.0, -18.0)]

        # 输出一些数据处理步骤中的异常信息, list或者写入log?
        # : 双面数据没有第四层 evt文件重复 膜厚没有7层等 问题...
        data_bug_list = []

        response = {
            "status": "OK",
            "oven_num": oven_num,
            "error_msg": str(data_bug_list),
            "data":
            {
                "original_lab_cure_similarity": original_lab_cure_similarity,
                "modified_lab_cure_similarity": modified_lab_cure_similarity,
                "cx_thick_material": cx_thick_material,
                "cx_base_thickness": cx_base_thickness,
                "cx_deta_thickness": cx_deta_thickness,
                "cx_modified_thickness": cx_modified_thickness,
                "cx_lab_curve": cx_lab_curve,
                "cc_thick_material": cc_thick_material,
                "cc_base_thickness": cc_base_thickness,
                "cc_deta_thickness": cc_deta_thickness,
                "cc_modified_thickness": cc_modified_thickness,
                "cc_lab_curve": cc_lab_curve,
                "modified_lab": modified_LAB,
                "double_lab_curve": double_lab_curve,
            }
        }

        return response


if __name__ == "__main__":
    
    #init model
    model_path = "./xxx"
    m = ZeissModel(model_path)

    # 输入文件位置
    data_dir = r'./csvs_dir'

    # 算法输出结果 和 数据异常信息
    res = m.run(data_dir)
    print(res, )

    # r = json.loads(res)

    # for item in r:
    #     print(item, r[item])