<template>
    <div class="wrap">
      <div class="header"></div>
      <div class="content">
        <div class="content-left">
          <div class="img-wrap">
            <img v-if="imageUrl" :src="imageUrl" alt="输入图片">
            <a-upload
              v-else
              style="margin:20px"
              list-type="picture-card"
              :file-list="fileList"
              :show-upload-list="false"
              :before-upload="beforeUpload"
              @preview="handlePreview"
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
          <div class="img-wrap">
            <img v-if="reducurl" :src="reducurl" alt="输出图片">
          </div>
          <a-button
            class="button"
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
export default {
  name: 'Home',
  data () {
    return {
      previewVisible: false,
      previewImage: '',
      imageUrl: '',
      fileList: [],
      uploading: false,
      reducurl:''
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
    async handlePreview(file) {
    },
    handleCancel() {
      this.previewVisible = false;
    },
    handleReset(){
      this.fileList = [];
      this.imageUrl = '';
      this.reducurl = '';
    },
    handleUpload() {
      const  fileList = this.fileList;
      let formData = new FormData(); 
      formData.append('pic', fileList[0]);
      console.log(this.fileList)
      console.log(formData)
      this.uploading = true;

      // this.$httpPost('http://localhost:8990/reduction',{'pic':fileList[0]},{"headers":{"Content-Type" : "multipart/form-data"}})
      // .then(({code,data,msg}) => {
      //   if (code === "0"){
      //     this.uploading = false;
      //     this.reducurl = data;
      //   } else {
      //     this.uploading = false;
      //     this.$message.error(msg);
      //   }
      // })
      // .catch(() => {
      //   this.$message.error("请求失败");
      //   this.uploading =false;
      //   })

      this.axios.post(this.$server+'/reduction',formData)
      .then((res) => {
        console.log(res)
        if (res.data.code === "000000"){
          this.uploading = false;
          // this.imageUrl = res.data.data.in;
          this.reducurl = res.data.data.out;
        } else {
          this.uploading = false;
          this.$message.error(res.data.msg);
        }
        console.log("111")
        }) .catch((err) => {
          this.$message.error("请求失败");
          this.uploading =false;
        });

      // reqwest({
      //   url: 'http://localhost:8990/reduction',
      //   method: 'post',
      //   processData: false,
      //   data: formData,
      //   success: ({code,data,msg}) => {
      //     if (code === "0"){
      //     this.uploading = false;
      //     this.reducurl = data;
      //   } else {
      //     this.uploading = false;
      //     this.$message.error(msg);
      //   }
      //   },
      //   error: () => {
      //     this.$message.error("请求失败");
      //   this.uploading =false;
      //   },
      // });

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
  width: 400px;
  min-height: 65vh;
  margin: 50px;
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
  width: 400px;
  min-height: 65vh;
  margin: 50px 50px 50px 0px;
  border-radius: 10px;
  border: 2px solid rgba(152, 171, 233, 0.5);
}
.img-wrap{
  margin: 30px;
  order: 1;
  height: 256px;
  width: 256px;
}
.button{
  order: 2;
  margin: 50px;
}
.description{
  order: 3;
}
img{
  margin: 30px 0;
  max-height: 256px;
  max-height: 256px;  
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