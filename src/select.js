var currentSimList = ["flood", "snow", "smog"];
var currentSim = "flood";
var currentSimId = 0;
var currentIntList = ["albedo", "depth", "normal", "visibility", "shading", "semantic"];
var currentInt = "albedo";
var currentIntId = 0;
var currentSceneList = ["6040", "1538"];
var currentScene = "6040";
var currentSceneId = 0;
var currentMethodList = ["ours", "GAN", "Diffusion", "Swap", "Sty"];
var currentMethod = "ours";

// vid = document.getElementById("showoff-img");
// vid.playbackRate = 0.40;	

// function ChangeDataset(){
//     var dataset = document.getElementById("dataset-select").value.split(" ")[0];
//     var frame_total = parseInt(document.getElementById("dataset-select").value.split(" ")[1]);
//     document.getElementById("frame-total").innerHTML = frame_total;
//     document.getElementById("frame-idx-input").value = 1;
//     document.getElementById("frame-idx-input").max = frame_total;
//     ChangeFrame();
// }

// function ChangeFrame(){
//     // var dataset = document.getElementById("dataset-select").value.split(" ")[0];
//     // var frame_idx = parseInt(document.getElementById("frame-idx-input").value);
//     // document.getElementById("frame-idx").innerHTML = ("00" + frame_idx).slice(-2);
//     console.log(currentSim);
//     console.log(currentScene);
//     document.getElementById("simulation_video").src = "src/ours/"+ currentSim + '/' + currentScene + '.mp4';

//     // document.getElementById("showoff-label").src = "visual_comparison/label/" +dataset +'_' +("00" + frame_idx).slice(-2) + '.png';

//     // vid = document.getElementById("simulation_video");
//     // vid.playbackRate = 1;	
// }

// function NextFrame(){
//     var frame_total = parseInt(document.getElementById("dataset-select").value.split(" ")[1]);
//     var frame_idx = parseInt(document.getElementById("frame-idx-input").value);
//     if(frame_idx < frame_total){
//         document.getElementById("frame-idx-input").value = frame_idx + 1;
//         ChangeFrame();
//     }
// }

// function PrevFrame(){
//     var frame_idx = parseInt(document.getElementById("frame-idx-input").value);
//     if(frame_idx > 1){
//         document.getElementById("frame-idx-input").value = frame_idx - 1;
//         ChangeFrame();
//     }
// }

function ChangeSim(idx){
    var li_list = document.getElementById("sim-view-ul").children;
    console.log(idx);
    console.log(li_list);
    for(i = 0; i < li_list.length; i++){
        li_list[i].className = "";
    }
    li_list[idx].className = "active";

    currentSim = currentSimList[idx];
    currentSimId = idx;

    var m_list = document.getElementById("method-view-ul").children;
    m_list[0] = "active";
    for (i=1;i<m_list.length;i++){
        m_list[i].className = "disabled";
        console.log(m_list[i].children[0].onclick)
        m_list[i].children[0].onclick = "";
    }
    console.log("###")
    for (i=0;i<m_list.length;i++){
        console.log(m_list[i].children[0].onclick)
    }
    if(currentSceneId<5){
        if(idx==0){
            m_list[1].className = "";
            m_list[1].children[0].onclick = function func(){ChangeMethod(1);}
            m_list[2].className = "";
            m_list[2].children[0].onclick = function func(){ChangeMethod(2);}
        }
        else if(idx==1){
            m_list[3].className = "";
            m_list[3].children[0].onclick = function func(){ChangeMethod(3);}
            m_list[4].className = "";
            m_list[4].children[0].onclick = function func(){ChangeMethod(4);}
        }
        else if(idx==2){
            m_list[1].className = "";
            m_list[1].children[0].onclick = function func(){ChangeMethod(1);}
            m_list[4].className = "";
            m_list[4].children[0].onclick = function func(){ChangeMethod(4);}
        }
    }

    document.getElementById("simulation_video").src = "src/ours/"+ currentSim + '/' + currentScene + '.mp4';
    ChangeMethod(0);
}

function ChangeInt(idx){
    var li_list = document.getElementById("sim-view-ul").children;
    console.log(idx);
    console.log(li_list);
    for(i = 0; i < li_list.length; i++){
        li_list[i].className = "";
    }
    li_list[idx].className = "active";

    currentInt = currentIntList[idx];
    currentIntId = idx;

    document.getElementById("simulation_video").src = "src/decomposition/"+ currentScene + '/rgb_' + currentInt + '.mp4';
}

function ChangeScene(idx){
    var li_list = document.getElementById("scene-view-ul").children;
    console.log(idx);
    console.log(li_list);

    for(i = 0; i < li_list.length; i++){
        li_list[i].className = "";
    }
    li_list[idx].className = "active";
    
    currentScene = currentSceneList[idx];
    currentSceneId = idx;

    let video = document.getElementById("simulation_video")
    let container = video.parentNode
    video.src = "src/decomposition/"+ currentScene + '/rgb_' + currentInt + '.mp4';

    container.style = "width: 100%; opacity: 0;"
    setTimeout(()=>{
        container.style = "width: 100%; opacity: 1;"
        video.load();
    }, 1000)

}

function ChangeMethod(idx){
    var li_list = document.getElementById("method-view-ul").children;
    console.log(idx);
    console.log(li_list);
    for(i = 0; i < li_list.length; i++){
        if (li_list[i].className === "disabled"){
            continue
        }
        li_list[i].className = "";
    }
    li_list[idx].className = "active";
    currentMethod = currentMethodList[idx]
    document.getElementById("baseline").src = "src/" + currentMethod +'/'+ currentSim +'/'+ currentScene + '.mp4';
    if(idx == 0){
        document.getElementById("ours").style="width:100%";
        document.getElementById("baseline").style="width:0%";
    }
    else if(idx >= 1){
        document.getElementById("ours").style="width:0%";
        document.getElementById("baseline").style="width:100%";
    }
}