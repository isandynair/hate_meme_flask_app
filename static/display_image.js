var loadFile = function (event) {
    var meme_image = document.getElementById('display_img');
    meme_image.src = URL.createObjectURL(event.target.files[0]);
};