<template>
    <div class="wrap">
      <div class="header"></div>
      <div class="content">
        <div class="content-left">
          <div class="img-wrap">
            <img class="img-left" v-if="imageUrl" :src="imageUrl" alt="输入图片">
            <a-upload
              v-else
              style="margin:20px"
              list-type="picture-card"
              :file-list="fileList"
              :show-upload-list="false"
              :before-upload="beforeUpload"
            >
              <div v-if="fileList.length<1" class="upload-conent">
                <a-icon :type="uploading ? 'loading' : 'plus'" style="fontSize:100px"/>
                <div class="ant-upload-text" style="fontSize:20px">
                  Upload
                </div>
              </div>
            </a-upload>
          </div>
          <a-button
          class="button"
          type="primary"
          :disabled="fileList.length === 0"
          :loading="uploading"
          @click="handleUpload"
        >
          {{ uploading ? 'Uploading' : 'Start Upload' }}
        </a-button>
        <div class="description">
          <p>文件大小不要超过2M！</p>
          <p>当文件分辨率大于256*256时，将会被压缩</p>
        </div>
        </div>
        <div class="content-right">
          <div class="img-wrap-right">
            <div
              class="img-right"
              v-for="(topn,index) in top5"
              :key="index"
              :style="{backgroundImage: 'url(' + topn.backUrl + ')', backgroundSize:'contain'}"
              @click="searchClass(topn.herf)"
            >
              <div class="tip">
                <div>{{topn.label}}</div>
                <div v-if="topn.accurracy">{{topn.accurracy}}</div>
              </div>
            </div>
            <div 
              class="img-right"
              :style="{backgroundImage: 'url(' + input_box.URL + ')', backgroundSize:'contain'}" >
              <a-form-model v-if="finished" layout="inline" :model="form" @submit="handleSubmit" @submit.native.prevent style="margin:40px 10px">
                <a-form-model-item>
                  <a-input v-model="form.input_name" placeholder="请输入物种名">
                    <a-icon slot="prefix" type="smile" style="color:rgba(0,0,0,.25)" />
                  </a-input>
                </a-form-model-item>
                <a-form-model-item>
                  <a-button
                    type="primary"
                    html-type="submit"
                    :disabled="form.input_name === ''"
                  >
                    确定
                  </a-button>
                </a-form-model-item>
              </a-form-model>
              <div class="tip">
                {{ finished ? '都不是，我知道正确答案！' : 'unknown' }}
              </div>
            </div>
            
          </div>
          <a-button
            class="button-right"
            type="primary"
            :disabled="fileList.length === 0"
            :loading="uploading"
            @click="handleReset"
          >
            {{ uploading ? 'Waiting' : 'Reset' }}
          </a-button>
          <div class="description">
            <p>点击reset清除数据</p>
            <p>程序长时间未响应不要担心，是正常现象</p>
          </div>
        </div>
      </div>
      <div class="floor">
      </div>
    </div>
</template>
<script>
function getBase64(img, callback) {
  const reader = new FileReader();
  reader.addEventListener('load', () => callback(reader.result));
  reader.readAsDataURL(img);
};
var top5 = [
  {
    key:1,
    label:'unknown',
    accurracy:"",
    backUrl:require(`@/assets/unknown.jpg`),
    herf:''
  },
  {
    key:2,
    label:'unknown',
    accurracy:"",
    backUrl:require(`@/assets/unknown.jpg`),
    herf:''
  },
    {
    key:3,
    label:'unknown',
    accurracy:"",
    backUrl:require(`@/assets/unknown.jpg`),
    herf:''
  },
    {
    key:4,
    label:'unknown',
    accurracy:"",
    backUrl:require(`@/assets/unknown.jpg`),
    herf:''
  },
    {
    key:5,
    label:'unknown',
    accurracy:"",
    backUrl:require(`@/assets/unknown.jpg`),
    herf:''
  }
];
var allInfo = {
  basset_hound:{
    background:require(`@/assets/kinds/basset_hound_19.jpg`),
    herf:'https://baike.baidu.com/item/%E5%B7%B4%E5%90%89%E5%BA%A6%E7%8C%8E%E7%8A%AC'},
  Bengal:{
    background:require(`@/assets/kinds/Bengal_79.jpg`),
    herf:'https://baike.baidu.com/item/%E5%AD%9F%E5%8A%A0%E6%8B%89%E7%8C%AB/4631122?fr=aladdin'},
  Birman:{
    background:require(`@/assets/kinds/Birman_99.jpg`),
    herf:'https://baike.baidu.com/item/%E4%BC%AF%E6%9B%BC%E7%8C%AB'},
  chihuahua:{
    background:require(`@/assets/kinds/chihuahua_9.jpg`),
    herf:'https://baike.baidu.com/item/%E5%90%89%E5%A8%83%E5%A8%83/124178'},
  chimpanzee:{
    background:require(`@/assets/kinds/chimpanzee_11.jpg`),
    herf:'https://baike.baidu.com/item/%E9%BB%91%E7%8C%A9%E7%8C%A9/50071'},
  dragonfly:{
    background:require(`@/assets/kinds/dragonfly_3.jpg`),
    herf:'https://baike.baidu.com/item/%E8%9C%BB%E8%9C%93/962'},
  elephant:{
    background:require(`@/assets/kinds/elephant_63.jpg`),
    herf:'https://baike.baidu.com/item/%E5%A4%A7%E8%B1%A1/229'},
  goat:{
    background:require(`@/assets/kinds/goat_278.jpg`),
    herf:'https://baike.baidu.com/item/%E7%BE%8A/1947'},
  hedgehog:{
    background:require(`@/assets/kinds/hedgehog_23.jpg`),
    herf:'https://baike.baidu.com/item/%E5%88%BA%E7%8C%AC/37604'},
  horse:{
    background:require(`@/assets/kinds/horse_29.jpg`),
    herf:'https://baike.baidu.com/item/%E9%A9%AC/7204564'},
  Kestrel:{
    background:require(`@/assets/kinds/Kestrel_62.jpg`),
    herf:'https://baike.baidu.com/item/%E7%BA%A2%E9%9A%BC'},
  lesser_panda:{
    background:require(`@/assets/kinds/lesser_panda_507.jpg`),
    herf:'https://baike.baidu.com/item/%E5%B0%8F%E7%86%8A%E7%8C%AB/22379'},
  macow:{
    background:require(`@/assets/kinds/macow_16.jpg`),
    herf:'https://baike.baidu.com/item/%E9%87%91%E5%88%9A%E9%B9%A6%E9%B9%89'},
  manchurian_tiger:{
    background:require(`@/assets/kinds/manchurian_tiger_20.jpg`),
    herf:'https://baike.baidu.com/item/%E8%A5%BF%E4%BC%AF%E5%88%A9%E4%BA%9A%E8%99%8E/5467196?fromtitle=%E4%B8%9C%E5%8C%97%E8%99%8E&fromid=51058&fr=aladdin'},
  panda:{
    background:require(`@/assets/kinds/panda_39.jpg`),
    herf:'https://baike.baidu.com/item/%E5%A4%A7%E7%86%8A%E7%8C%AB/34935?fromtitle=%E7%86%8A%E7%8C%AB&fromid=162918'},
  pug:{
    background:require(`@/assets/kinds/pug_197.jpg`),
    herf:'https://baike.baidu.com/item/%E5%B7%B4%E5%93%A5%E7%8A%AC?fromtitle=%E5%85%AB%E5%93%A5%E7%8A%AC&fromid=125973'},
  samoyed:{
    background:require(`@/assets/kinds/samoyed_57.jpg`),
    herf:'https://baike.baidu.com/item/%E8%90%A8%E6%91%A9%E8%80%B6%E7%8A%AC/6397?fromtitle=%E8%90%A8%E6%91%A9%E8%80%B6&fromid=123391'},
  seals:{
    background:require(`@/assets/kinds/seals_8.jpg`),
    herf:'https://baike.baidu.com/item/%E6%B5%B7%E8%B1%B9/793253'},
  spider:{
    background:require(`@/assets/kinds/spider_99.jpg`),
    herf:'https://baike.baidu.com/item/%E8%9C%98%E8%9B%9B/6152'},
  swan:{
    background:require(`@/assets/kinds/swan_1.jpg`),
    herf:'https://baike.baidu.com/item/%E5%A4%A9%E9%B9%85/53209'}
}
var input_box = {
  URL:require(`@/assets/unknown.jpg`)
}

export default {
  name: 'Home',
  data () {
    return {
      top5,
      form:{input_name:''},
      allInfo,
      input_box,
      imageUrl: '',
      fileList: [],
      uploading: false,
      finished:false,
      reducurl:'',
    }
  },
  methods:{
    beforeUpload(file) {
      const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png';
      if (!isJpgOrPng) {
        this.$message.error('You can only upload JPG file!');
      }
      const isLt2M = file.size / 1024 / 1024 < 2;
      if (!isLt2M) {
        this.$message.error('Image must smaller than 2MB!');
      }
      this.fileList = [...this.fileList, file];
      console.log(this.fileList);
      getBase64(file, imageUrl => {
          this.imageUrl = imageUrl;
        });
      return false;
    },
    handleReset(){
      this.fileList = [];
      this.imageUrl = '';
      this.finished = false;
      for(let i = 0; i < this.top5.length; i++){
        this.top5[i].label = 'unknown';
        this.top5[i].accurracy = '';
        this.top5[i].herf = '';
        this.top5[i].backUrl = require(`@/assets/unknown.jpg`);
      }
    },
    handleSubmit(){
      const  fileList = this.fileList;
      let formData = new FormData();
      formData.append('pic', fileList[0]);
      formData.append('class',this.form.input_name)
      this.axios.post('http://localhost:8990/dbAdd',formData)
      .then((res) => {
        if (res.data.code === "000000"){
          this.$message.success("成功！")
        }else{
          this.$message.error("上传失败！")
        }}).catch((err) => {
          this.$message.error("请求失败")
        });
    },
    searchClass(url){
      if(this.finished){
        window.location.href = url;
      }
    },
    handleUpload() {
      const  fileList = this.fileList;
      let formData = new FormData(); 
      formData.append('pic', fileList[0]);
      console.log(this.fileList)
      console.log(formData)
      this.uploading = true;

      this.axios.post(this.$server+'/classify',formData)
      .then((res) => {
        console.log(res)
        if (res.data.code === "000000"){
          this.uploading = false;
          var backdata = res.data.data
          console.log(backdata)
          for(let i = 0; i < backdata.length; i++){
            this.top5[i].label = backdata[i].label;
            this.top5[i].accurracy = (backdata[i].accurracy*100).toFixed(2).toString()+"%";
            this.top5[i].herf = this.allInfo[backdata[i].label].herf;
            this.top5[i].backUrl = this.allInfo[backdata[i].label].background;
          }
          this.finished = true;
          console.log(this.top5)
        } else {
          this.uploading = false;
          this.$message.error(res.data.msg);
        }
        console.log("111")
        }) .catch((err) => {
          this.$message.error("请求失败");
          this.uploading =false;
        });
    },
  }

}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.wrap{
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: space-around;
  align-items: center;
  overflow: auto;
  background: linear-gradient(#fbfdff,#e7f3ff);
  padding: 40px 0;
}
.content{
  order:1;
  height: 79vh;
  width: 1000px;
  min-width: 1000px;
  margin: auto;
  background: #fff;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
}
.floor{
  order: 2;
  width:1000px
}
.content-left{
  order: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-self: flex-end;
  height: auto;
  width: 250px;
  min-height: 76vh;
  margin: 20px 10px 20px 20px;
  border-radius: 10px;
  border: 2px solid rgba(152, 171, 233, 0.5);

}
.content-right{
  order: 2;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  height: auto;
  width: 700px;
  min-height: 76vh;
  margin: 20px 20px 20px 0px;
  border-radius: 10px;
  border: 2px solid rgba(152, 171, 233, 0.5);
}
.img-wrap{
  margin: 30px 0px;
  order: 1;
  height: 250px;
  width: 250px;
  display: flex;
  flex-direction: column;
  justify-content: space-around;
}
.img-wrap-right{
  margin: 30px 0px;
  order: 1;
  height: 370px;
  width: 600px;
  display: flex;
  flex-direction: row;
  justify-content: space-around;
  flex-wrap:wrap;
}
.img-right{
  width: 180px;
  height: 180px;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
}
.img-right:hover{
  box-shadow: 10px 10px 5px #888888;
  cursor: pointer;
}
.tip{
  background:  rgba(255, 255, 255, 0.5);
  height: 10hv;
  display: flex;
  flex-direction: row;
  justify-content: space-around;
}
.button{
  order: 2;
  margin: 50px;
}
.button-right{
  order: 2;
  margin: 20px 50px;
}
.description{
  order: 3;
}
.img-left{
  max-height: 230px;
  max-width: 230px;
}
img{
  margin: auto;
  max-height: 200px;
  max-height: 200px;  
}
.upload-conent{
  height: 200px;
  width: 200px;
  margin: auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-around;
}
</style>