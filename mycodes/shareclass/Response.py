from StandardCode import StandardCode
import json


class Response():
    @staticmethod
    def error1(code, msg):
        return {'code': code, 'msg': msg, 'data': None}

    @staticmethod
    def ok(data):
        return {'code': StandardCode.SUCCESS.code, 'msg': StandardCode.SUCCESS.msg, 'data': data}

    @staticmethod
    def ok1(msg, data):
        return {'code': StandardCode.SUCCESS.code, 'msg': msg, 'data': data}

    @staticmethod
    def OK():
        return Response.ok(True)

    @staticmethod
    def error(msg):
        return {'code': "500", 'msg': msg, 'data': None}
