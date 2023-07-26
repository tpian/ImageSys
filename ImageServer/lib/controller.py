import flask
from flask import request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from ERROR import ERROR
from shareclass import Response
import subprocess
import traceback
import tempfile
import pymysql
import datetime

# 配置数据库
conn = pymysql.connect(host='localhost', port=3306, user="root", passwd="a123456", charset="utf8", db="images")
if conn:
    print("连接数据库成功！")
cursor = conn.cursor()


# 配置服务器
server3 = flask.Flask(__name__)
server3.config['UPLOAD_FOLDER_C'] = 'tempfile/classification/origin'
server3.config['UPLOAD_FOLDER_R'] = 'tempfile/reduction/origin'
server3.config['UPLOAD_FOLDER_DB'] = '../newImgs'

CORS(server3, supports_credentials=True)

@server3.route('/dbAdd', methods=['POST'])
def dbAdd():
    f = request.files['pic']
    subdir = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
    filename, ext = os.path.splitext(secure_filename(f.filename))
    origin_path = os.path.join(server3.config['UPLOAD_FOLDER_DB'], filename + "_" + subdir + ext)
    f.save(origin_path)
    classname = request.form['class']
    idi = id(origin_path)
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sql = "INSERT INTO `temp_images`(`id`, `image_path`, `image_class`, `create_time`) VALUES (%s, %s, %s, %s)"
    try:
        cursor.execute(sql, (idi, origin_path, classname, date))
        conn.commit()
        return flask.jsonify(Response.Response.OK())
    except:
        conn.rollback()  # 发生错误时回滚
        return flask.jsonify(ERROR.EXCEPTION)

@server3.route('/classify', methods=['post'])
def classifyImg():
    f = request.files['pic']
    origin_path = os.path.join(server3.config['UPLOAD_FOLDER_C'], secure_filename(f.filename))
    f.save(origin_path)
    try:
        cmd = 'python classify_interface.py'
        out_temp = tempfile.TemporaryFile(mode='w+')
        fileno = out_temp.fileno()
        obj = subprocess.Popen(cmd, stdout=fileno, stderr=fileno, shell=True)
        obj.wait()
        out_temp.seek(0)
        rt = out_temp.read()
        rt_list = rt.strip().split('\n')
    except Exception as e:
        print(traceback.format_exc())
    finally:
        if out_temp:
            out_temp.close()
    f = open('tempfile/classification/output/predict.txt', 'r')
    dict = []
    i = 0
    for line in f.readlines():
        i = i + 1
        par = line.strip().split(':')
        label = par[0]
        accurracy = float(par[1])
        dict.append({'label': label, 'accurracy': accurracy})
        if i == 5:
            break
    return flask.jsonify(Response.Response.ok(dict))


@server3.route('/reduction', methods=['POST'])
def reductImg():
    f = request.files['pic']
    origin_path = os.path.join(server3.config['UPLOAD_FOLDER_R'], secure_filename(f.filename))
    f.save(origin_path)
    try:
        cmd = 'python reduction_interface.py --filename ' + request.files['pic'].filename
        out_temp = tempfile.TemporaryFile(mode='w+')
        fileno = out_temp.fileno()
        obj = subprocess.Popen(cmd, stdout=fileno, stderr=fileno, shell=True)
        obj.wait()
        out_temp.seek(0)
        rt = out_temp.read()
        rt_list = rt.strip().split('\n')
    except Exception as e:
        print(traceback.format_exc())
    finally:
        if out_temp:
            out_temp.close()

    f = open('tempfile/reduction/imagebase64.txt', 'r')
    base64strout = f.read()
    f.close()
    f = open('tempfile/reduction/originbase64.txt', 'r')
    base64strin = f.read()
    f.close()
    return flask.jsonify(Response.Response.ok({'in': base64strin, 'out': base64strout}))





server3.run(port=8990, debug=True, host='0.0.0.0')
