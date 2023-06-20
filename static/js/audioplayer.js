// Get the file input elements
const fileInput1 = document.getElementById('file1');
const fileInput2 = document.getElementById('file2');

// Add event listeners to the file input fields
fileInput1.addEventListener('change', playAudio1);
fileInput2.addEventListener('change', playAudio2);

// Function to play audio for file1
function playAudio1() {
  const file1 = fileInput1.files[0];
  const url1 = URL.createObjectURL(file1);
  const audio1 = document.getElementById('audio1');
  audio1.src = url1;
  audio1.play();
}

// Function to play audio for file2
function playAudio2() {
  const file2 = fileInput2.files[0];
  const url2 = URL.createObjectURL(file2);
  const audio2 = document.getElementById('audio2');
  audio2.src = url2;
  audio2.play();
}

function submitFiles() {
    // Get the selected files
    const file1 = document.getElementById('file1').files[0];
    const file2 = document.getElementById('file2').files[0];
  
    // Perform some processing with the files using your ML model
    // Replace this with your actual logic to get the result from the model
    const result = performModelProcessing(file1, file2);
  
    // Show the result in a popup
    alert('Model Result: ' + result);
  }
  
  // Placeholder function for model processing (replace with your actual model logic)
  function performModelProcessing(file1, file2) {
    // Perform processing with the files and return the result
    return 'Dummy Model Result';
  }


  function submitFile2() {
    // Get the selected file
    const file1 = document.getElementById('file1').files[0];
    
    // Create a FileReader object to read the file
    const reader = new FileReader();
    
    // When the file is loaded
    reader.onload = function(event) {
        // Get the file data
        const fileData = event.target.result;
        
        // Update the src attribute of the audio source
        const audioElement = document.getElementById('audio2');
        const audioSource = audioElement.querySelector('source');
        audioSource.src = fileData;
        
        // Reload the audio element to update the source
        audioElement.load();
    };
    
    // Read the selected file as data URL
    reader.readAsDataURL(file1);
}





//---------------------------------------------------------------------------------------------------------------


// Variables for the sound wave visualization
const soundWaveCanvas = document.getElementById('soundWaveCanvas');
const soundWaveCtx = soundWaveCanvas.getContext('2d');
const soundWaveContainer = document.querySelector('.sound-wave-container');
let animationId;
let isPlaying = false;
let audioContext;
let analyser;
let dataArray;
let bufferLength;

// Function to initialize the audio context and analyser
function initAudio() {
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;

  // Get the buffer length once during initialization
  bufferLength = analyser.frequencyBinCount;
  dataArray = new Uint8Array(bufferLength);
}

// Function to draw the sound wave visualization
function drawSoundWave() {
  const audio1 = document.getElementById('audio1');
  const audio2 = document.getElementById('audio2');

  const audioElement = audio1.paused ? audio2 : audio1; // Get the currently playing audio element

  soundWaveCtx.fillStyle = '#fff';
  soundWaveCtx.strokeStyle = '#fff';
  soundWaveCtx.lineWidth = 4;

  function draw() {
    animationId = requestAnimationFrame(draw);

    const width = soundWaveCanvas.width;
    const height = soundWaveCanvas.height;

    analyser.getByteTimeDomainData(dataArray);

    soundWaveCtx.clearRect(0, 0, width, height);

    soundWaveCtx.beginPath();

    const sliceWidth = width * 1.0 / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = v * height / 2;

      if (i === 0) {
        soundWaveCtx.moveTo(x, y);
      } else {
        soundWaveCtx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    soundWaveCtx.lineTo(width, height / 2);
    soundWaveCtx.stroke();
  }

  draw();
}

// Event listener for audio playback start
document.addEventListener('play', function (e) {
  if (!isPlaying) {
    isPlaying = true;
    soundWaveContainer.style.display = 'block';
    initAudio();
    const source = audioContext.createMediaElementSource(e.target);
    source.connect(analyser).connect(audioContext.destination);
    drawSoundWave();
  }
}, true);

// Event listener for audio playback pause
document.addEventListener('pause', function (e) {
  if (isPlaying) {
    isPlaying = false;
    cancelAnimationFrame(animationId);
  }
}, true);

//---------------------------------------------------RECORD Button--------------------------------------------------------
// Variables for recording audio
let isRecording1 = false;
let isRecording2 = false;
let mediaRecorder1;
let mediaRecorder2;
let chunks1 = [];
let chunks2 = [];

function toggleRecording(playerNumber) {
  if (playerNumber === 1) {
    if (!isRecording1) {
      startRecording(1);
    } else {
      stopRecording(1);
    }
  } else if (playerNumber === 2) {
    if (!isRecording2) {
      startRecording(2);
    } else {
      stopRecording(2);
    }
  }
}

function startRecording(playerNumber) {
  const constraints = { audio: true };
  
  navigator.mediaDevices.getUserMedia(constraints)
    .then(function (stream) {
      if (playerNumber === 1) {
        mediaRecorder1 = new MediaRecorder(stream);
        mediaRecorder1.start();

        mediaRecorder1.addEventListener('dataavailable', function (event) {
          chunks1.push(event.data);
        });
      } else if (playerNumber === 2) {
        mediaRecorder2 = new MediaRecorder(stream);
        mediaRecorder2.start();

        mediaRecorder2.addEventListener('dataavailable', function (event) {
          chunks2.push(event.data);
        });
      }
      
      if (playerNumber === 1) {
        isRecording1 = true;
      } else if (playerNumber === 2) {
        isRecording2 = true;
      }
    })
    .catch(function (error) {
      console.error('Error accessing the microphone:', error);
    });
}

function stopRecording(playerNumber) {
  if (playerNumber === 1) {
    mediaRecorder1.stop();

    mediaRecorder1.addEventListener('stop', function () {
      const blob = new Blob(chunks1, { type: 'audio/webm' });
      chunks1 = [];

      const audioURL = URL.createObjectURL(blob);
      const audioPlayer = document.getElementById('audio1');
      audioPlayer.src = audioURL;
      audioPlayer.load();
    });

    isRecording1 = false;
  } else if (playerNumber === 2) {
    mediaRecorder2.stop();

    mediaRecorder2.addEventListener('stop', function () {
      const blob = new Blob(chunks2, { type: 'audio/webm' });
      chunks2 = [];

      const audioURL = URL.createObjectURL(blob);
      const audioPlayer = document.getElementById('audio2');
      audioPlayer.src = audioURL;
      audioPlayer.load();
    });

    isRecording2 = false;
  }
}
