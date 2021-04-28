import flask


server1 = flask.Flask(__name__)


@server1.route('/classify', methods=['post'])
def classifyimg():
    # 加载classify模型的验证
    return flask.jsonify()


@server1.route('/reduction', methods=['post'])
def reductionimg():
    # 加载reduction模型的验证
    return flask.jsonify()

@server1.route('/appendDB', methods=['post'])
def appendDB():
    # 扩充classify模型数据库
    return flask.jsonify()

server1.run(port=8999, debug=True, host='0.0.0.0')
