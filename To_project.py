# encoding:utf-8
from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource

# 初始化app、api
app = Flask(__name__)
api = Api(app)

LISTS = [
    {'parameter': '首页'},
    {'parameter': '登录'},
    {'parameter': '后台'}
]


# /LISTS/<list_id>（url参数），判断输入的参数值列表LISTS下标越界，越界则退出
def abort_if_list_doesnt_exist(list_id):
    try:
        LISTS[list_id]
    except IndexError:
        abort(404, message="输入的值，不在范围内")


'''
add_argument('per_page', type=int, location='args') str
add_argument中通过指定参数名、参数类型、参数获取方式来获取参数对象并支持做合法性校验
第一个参数是需要获取的参数的名称
参数type: 参数指的类型， 如果参数中可能包含中文需要使用six.text_type. 或直接不指定type
参数location: 获取参数的方式，可选的有args(url中获取)、json(json类型的)、form(表单方式提交)
参数required:是否必要，默认非必要提供  required=True(必须)
参数help:针对必要的参数，如果请求时没有提供，则会返回help中相应的信息
'''
parser = reqparse.RequestParser()
# 入参parameter，location='json'表示为入参为json格式
parser.add_argument('parameter', location='json')


# 路由类，函数get、post、put、delete等实现http请求方法
# url不带入参  /LISTS
class c_dictList(Resource):
    # 类型get，根据列表LISTS，处理，返回一个新的列表r_lists
    def get(self):
        r_lists = []
        for listV in LISTS:
            if listV:
                new_list = {}
                # LISTS列表存的是字典，遍历时为字典listV['parameter']，可获取字典值
                new_list['parameter'] = listV['parameter']
                # LISTS为列表，index可以查出对应下标值
                new_list['url'] = 'url/' + str(LISTS.index(listV))
                # LISTS列表中添加字典
                r_lists.append(new_list)
        return r_lists

    # 类型post，在列表LISTS后添加一个值，并返回列表值
    def post(self):
        args = parser.parse_args()
        list_id = len(LISTS)
        # args['parameter']，入参
        LISTS.append({'parameter': args['parameter']})
        return LISTS, 201


# 路由类，函数get、post、put、delete等实现http请求方法
# url带入参  /LISTS/<list_id>
class c_dict(Resource):
    # 根据输入url入参值作为LISTS的下标，返回该值
    def get(self, list_id):
        url_int = int(list_id)
        abort_if_list_doesnt_exist(url_int)
        return LISTS[url_int]

    # 根据输入url入参值作为LISTS的下标，修改该值，并返回列表值
    def put(self, list_id):
        url_int = int(list_id)
        args = parser.parse_args()
        # args['parameter']，入参
        parameter = {'parameter': args['parameter']}
        LISTS[url_int] = parameter
        return LISTS, 201

    # 根据输入url入参值作为LISTS的下标，删除该值
    def delete(self, list_id):
        url_int = int(list_id)
        abort_if_list_doesnt_exist(url_int)
        del LISTS[url_int]
        return '', 204


# 设置资源路由api.add_resource（类名，url路径）
# url，不带入参，如：http://127.0.0.1:8891/LISTS
api.add_resource(c_dictList, '/LISTS')
# url，带入参，<list_id>为变量值，如：http://127.0.0.1:8891/LISTS/1
api.add_resource(c_dict, '/LISTS/<list_id>')

if __name__ == '__main__':
    # 不设置ip、端口，默认：http://127.0.0.1:5000/
    app.run(debug=True)
    # 设置ip、端口
    # app.run(host="127.0.0.1", port=8891, debug=True)