window.HELP_IMPROVE_VIDEOJS = false;

// var INTERP_BASE = "https://homes.cs.washington.edu/~kpar/nerfies/interpolation/stacked";
// var NUM_INTERP_FRAMES = 240;

// var interp_images = [];
// function preloadInterpolationImages() {
//   for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
//     var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
//     interp_images[i] = new Image();
//     interp_images[i].src = path;
//   }
// }

// function setInterpolationImage(i) {
//   var image = interp_images[i];
//   image.ondragstart = function() { return false; };
//   image.oncontextmenu = function() { return false; };
//   $('#interpolation-image-wrapper').empty().append(image);
// }

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    // $('.results-carousel').slick({
    //   dots: true,
    //   infinite: true,
    //   speed: 300,
    //   slidesToShow: 1,
    //   autoplay: true,
    //   autoplaySpeed: 5000
    // });

    var options = {
      slidesToScroll: 1,
      slidesToShow: 3,
      loop: true,
      infinite: true,
      autoplay: false,
      autoplaySpeed: 3000,
      interval: 5000
    }

    // Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    // preloadInterpolationImages();

    // $('#interpolation-slider').on('input', function(event) {
    //   setInterpolationImage(this.value);
    // });
    // setInterpolationImage(0);
    // $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();
})


$(window).on("load", function(){
    // Reset gifs once everything is loaded to synchronize playback.
    $('.preload').attr('src', function(i, a){
        $(this).attr('src','').removeClass('preload').attr('src', a);
    });

    $('.author-portrait').each(function() {
      $(this).mouseover(function() {
          $(this).find('.depth').css('top', '-100%');
      });
      $(this).mouseout(function() {
          $(this).find('.depth').css('top', '0%');
      });
    });


    // const position = { x: 0, y: 0 }
    // const box = $('.hyper-space');
    // const cursor = $('.hyper-space-cursor');
    // interact('.hyper-space-cursor').draggable({
    //   listeners: {
    //     start (event) {
    //       console.log(event.type, event.target)
    //     },
    //     move (event) {
    //       position.x += event.dx
    //       position.y += event.dy

    //       event.target.style.transform =
    //         `translate(${position.x}px, ${position.y}px)`

    //       let childPos = cursor.offset();
    //       let parentPos = box.offset();
    //       let childSize = cursor.outerWidth();
    //       let point = {
    //           x: (childPos.left - parentPos.left),
    //           y: (childPos.top - parentPos.top)
    //       };
    //       point = {
    //         x: (point.x) / (box.innerWidth() - childSize),
    //         y: (point.y) / (box.innerHeight() - childSize)
    //       }
    //       updateHyperGrid(point);
    //     },
    //   },
    //   modifiers: [
    //     interact.modifiers.restrictRect({
    //       restriction: 'parent'
    //     })
    //   ]
    // });

});

Number.prototype.clamp = function(min, max) {
  return Math.min(Math.max(this, min), max);
};


// function updateHyperGrid(point) {
//   const n = 20 - 1;
//   let top = Math.round(n * point.y.clamp(0, 1)) * 100;
//   let left = Math.round(n * point.x.clamp(0, 1)) * 100;
//   $('.hyper-grid-rgb > img').css('left', -left + '%');
//   $('.hyper-grid-rgb > img').css('top', -top + '%');
// }