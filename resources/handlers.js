class ImageHandler {
  constructor(n_datasets, n_images, databtn_id, imgbtn_id, imgtag_id) {
    this.n_datasets = n_datasets;
    this.n_images = n_images;

    this.databtn_id = databtn_id;
    this.imgbtn_id = imgbtn_id;
    this.imgtag_id = imgtag_id;
  }

  select_image(j) {
    for (let i = 0; i < this.n_images; i++) {
      let v = document.getElementById(this.imgtag_id + i.toString());
      let b = document.getElementById(this.imgbtn_id + i.toString());
      if (i == j) {
        b.className = "on";
        v.style.display = "block";
      } else {
        b.className = "";
        v.style.display = "none";
      }
    }
  };

 select_dataset(j) {
    let dataset_name = document.getElementById(this.databtn_id + j.toString()).value;
    for (let i = 0; i < this.n_datasets; i++) {
      document.getElementById(this.databtn_id + i.toString()).className = (i == j ? "on" : "");
    }
    for (let i = 0; i < this.n_images; i++) {
      let image_name = document.getElementById(this.imgbtn_id + i.toString()).value;
      let v = document.getElementById(this.imgtag_id + i.toString());
      v.src = "images/" + dataset_name + "_" + image_name + ".png";
    }
  };

  register() {
    for (let i = 0; i < this.n_datasets; i++) {
      document.getElementById(this.databtn_id + i.toString()).addEventListener("click", function() { this.select_dataset(i); }.bind(this, i));
    }

    for (let i = 0; i < this.n_images; i++) {
      document.getElementById(this.imgbtn_id + i.toString()).addEventListener("click", function() { this.select_image(i); }.bind(this, i));
    }

    //this.select_image(0);
  }
}

class VideoHandler {
  constructor(n_datasets, n_videos, databtn_id, vidbtn_id, vidtag_id) {
    this.n_datasets = n_datasets;
    this.n_videos = n_videos;

    this.databtn_id = databtn_id;
    this.vidbtn_id = vidbtn_id;
    this.vidtag_id = vidtag_id;
  }

  get paused() {
    return document.getElementById(this.vidtag_id + "0").paused;
  }

  sync_video(e) {
    if (e === undefined) {
      return;
    }
    for (let i = 0; i < this.n_videos; i++) {
      let v = document.getElementById(this.vidtag_id + i.toString());
      if (v != e.currentTarget) {
        v.currentTime = e.currentTarget.currentTime;
      }
    }
  };

  play_video(e) {
    this.sync_video(e);
    for (let i = 0; i < this.n_videos; i++) {
      document.getElementById(this.vidtag_id + i.toString()).play();
    }
  };

  pause_video(e) {
    for (let i = 0; i < this.n_videos; i++) {
      document.getElementById(this.vidtag_id + i.toString()).pause();
    }
    this.sync_video(e);
  };

  select_video(j) {
    for (let i = 0; i < this.n_videos; i++) {
      let v = document.getElementById(this.vidtag_id + i.toString());
      let b = document.getElementById(this.vidbtn_id + i.toString());
      if (i == j) {
        b.className = "on";
        v.style.display = "block";
        v.addEventListener("play", this);
        v.addEventListener("pause", this);
        v.addEventListener("seeking", this);
        v.addEventListener("seeked", this);
        v.addEventListener("playing", this);
      } else {
        b.className = "";
        v.style.display = "none";
        v.removeEventListener("play", this);
        v.removeEventListener("pause", this);
        v.removeEventListener("seeking", this);
        v.removeEventListener("seeked", this);
        v.removeEventListener("playing", this);
      }
    }
  };

  handleEvent(e) {
    switch (e.type) {
      case "play": this.play_video(e); break;
      case "pause": this.pause_video(e); break;
      case "seeking": this.sync_video(e); break;
      case "seeked": this.sync_video(e); break;
      case "playing": this.sync_video(e); break;
    }
  }

  select_dataset(j) {
    let autoplay = !this.paused;
    let dataset_name = document.getElementById(this.databtn_id + j.toString()).value;
    for (let i = 0; i < this.n_datasets; i++) {
      document.getElementById(this.databtn_id + i.toString()).className = (i == j ? "on" : "");
    }
    for (let i = 0; i < this.n_videos; i++) {
      let video_name = document.getElementById(this.vidbtn_id + i.toString()).value;
      let v = document.getElementById(this.vidtag_id + i.toString());

      if (v.src.includes('.mov')) {
        v.src = "videos/" + dataset_name + "_" + video_name + ".mov";
      }
      else {
        v.src = "videos/" + dataset_name + "_" + video_name + ".mov";
      }
    }
    if (autoplay) {
      this.play_video();
    }
  };

  register() {
    for (let i = 0; i < this.n_datasets; i++) {
      document.getElementById(this.databtn_id + i.toString()).addEventListener("click", function() { this.select_dataset(i); }.bind(this, i));
    }

    for (let i = 0; i < this.n_videos; i++) {
      document.getElementById(this.vidbtn_id + i.toString()).addEventListener("click", function() { this.select_video(i); }.bind(this, i));
    }

    for (let i = 0; i < this.n_videos; i++) {
      document.getElementById(this.vidtag_id + i.toString()).muted = true;
    }

    //this.select_video(0);
  }
}


