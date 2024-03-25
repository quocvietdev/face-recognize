const express = require("express");
const faceapi = require("@vladmandic/face-api");
const mongoose = require("mongoose");
const { Canvas, Image } = require("canvas");
const canvas = require("canvas");
const fileUpload = require("express-fileupload");
faceapi.env.monkeyPatch({ Canvas, Image });
require('@tensorflow/tfjs-node')
const { performance } = require('perf_hooks');
var cors = require('cors')

const multer = require('multer');
const cloudinary = require('cloudinary').v2;

// Return "https" URLs by setting secure: true
cloudinary.config({
  secure: true
});

const app = express();
const uploadImage = async (imagePath) => {

  // Use the uploaded file's name as the asset's public ID and 
  // allow overwriting the asset with new versions
  const options = {
    use_filename: true,
    unique_filename: false,
    overwrite: true,
  };

  try {
    // Upload the image
    const result = await cloudinary.uploader.upload(imagePath, options);
    console.log(result);
    return result.url;
  } catch (error) {
    console.error(error);
  }
};
app.use(
  fileUpload({
    useTempFiles: true,
  })
);

app.use(
  cors({
    origin: '*',
    // Allow follow-up middleware to override this CORS for options
    preflightContinue: true,
  }),
);


async function LoadModels() {
  // Load the models
  // __dirname gives the root directory of the server
  await faceapi.nets.faceRecognitionNet.loadFromDisk(__dirname + "/models");
  await faceapi.nets.faceLandmark68Net.loadFromDisk(__dirname + "/models");
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(__dirname + "/models");
}
LoadModels();


const faceSchema = new mongoose.Schema({
  label: {
    type: String,
    required: true,
    unique: true,
  },
  descriptions: {
    type: Array,
    required: true,
  },
  image_url: {
    type: String,
    required: true,
    unique: true,
  },
});

const FaceModel = mongoose.model("Face", faceSchema);

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/'); // Uploads will be stored in the 'uploads' folder
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname); // Unique filename
  }
});

const upload = multer({ storage: storage });
async function uploadLabeledImages(images, label) {
  try {
    let counter = 0;
    const descriptions = [];
    // Loop through the images
   let data = await uploadImage(images[0])
   
      const img = await canvas.loadImage(images[0]);
      counter = 1 * 100;
      console.log(`Progress = ${counter}% ,${0}`);
      // Read each face and save the face descriptions in the descriptions array
      const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
     
     if(detections?.descriptor){
      descriptions.push(detections.descriptor);
      const createFace = new FaceModel({
        label: label,
        descriptions: descriptions,
        image_url:data
      });
     let dataTest = await createFace.save();
      return true;
     }else{
      return false;
     }
    
    
   

    // Create a new face document with the given label and save it in DB
    
  } catch (error) {
    console.log("error",error);
    return false;
  }
}

async function getDescriptorsFromDB(image) {
  // Get all the face data from mongodb and loop through each of them to read the data
  let faces = await FaceModel.find();

  let dataFaceRaw = [...faces]
  console.log("faces",faces)
  

  for (i = 0; i < faces.length; i++) {
    // Change the face data descriptors from Objects to Float32Array type
    for (j = 0; j < faces[i].descriptions.length; j++) {
      faces[i].descriptions[j] = new Float32Array(Object.values(faces[i]?.descriptions[j]));
    }
    // Turn the DB face docs to
    faces[i] = new faceapi.LabeledFaceDescriptors(faces[i].label, faces[i]?.descriptions);
  }

  // Load face matcher to find the matching face
  const faceMatcher = new faceapi.FaceMatcher(faces, 0.6);

  // Read the image using canvas or other method
  const img = await canvas.loadImage(image);
  let temp = faceapi.createCanvasFromMedia(img);
  // Process the image for the model
  const displaySize = { width: img.width, height: img.height };
  faceapi.matchDimensions(temp, displaySize);

  // Find matching faces
  const detections = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors();
  const resizedDetections = faceapi.resizeResults(detections, displaySize);
  const results = resizedDetections.map((d) => faceMatcher.findBestMatch(d.descriptor));
  if(results?.length > 0){
    let dataResult = dataFaceRaw?.find(x=>x?.label === results[0]?.label)
    return dataResult;
  }else{
    return results
  }

}

app.post("/post-face",async (req,res)=>{
    const File1 = req.files?.File1?.tempFilePath
    const label = req.body.label
    let result = await uploadLabeledImages([File1], label);
    if(result){
        res.json({message:"Face data stored successfully"})
    }else{
        res.json({message:"Your face cannot be analyzed, please try again."})
        
    }
})

app.post("/check-face", async (req, res) => {
  const start = performance.now();
  const File1 = req?.files?.File1?.tempFilePath;
  if(File1){
  let result = await getDescriptorsFromDB(File1);
  const end = performance.now();
    console.log(`Time taken to execute comparing function is ${end - start}ms.`);
  res.json({ result });
  }else{
    res.json({message:"Something went wrong, please try again."})
  }
});


// add your mongo key instead of the ***
mongoose
  .connect(
    "mongodb+srv://quoc:bcxstudio@cluster0.ebjzyoa.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
    {
      useNewUrlParser: true,
      useUnifiedTopology: true,
      useCreateIndex: true,
    }
  )
  .then(() => {
    app.listen(process.env.PORT || 3000);
    console.log("DB connected and server us running.");
  })
  .catch((err) => {
    console.log(err);
  });