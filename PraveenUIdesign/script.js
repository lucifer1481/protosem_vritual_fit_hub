// Start webcam stream
const video = document.getElementById('webcam');
navigator.mediaDevices
  .getUserMedia({ video: true, audio: false })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(err => {
    alert('Could not access webcam. Please allow permission.');
    console.error(err);
  });

// Click handler for outfit images
function selectCloth(name) {
  alert(`You selected: ${name}`);
}

// Share and Download placeholders
function share() {
  alert('Sharing feature coming soon!');
}

function download() {
  alert('Download feature coming soon!');
}
