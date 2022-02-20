// javascript logic for the watersip website
// HS, 16.09.2016

window.onload=initWebsite;

// date settings
//var cday = new Date(2018,2,18);
var cday = new Date();
var fdate = 0; // forecast time step
var member1 = 1;
var member2 = 1;
var zoom=false;
var kind = 1;
var bt = 0;

// treshold settings and names
var thresholdNames=new Array("5mm","10mm");
var thresholds=new Array("th05","th10");
var threshold=0;
var captions=new Array(2);
captions[0]="";
captions[1]="";
captions[2]="";

// functions
function getKind()
{
  switch (kind) {
    case 1:
      return "_iwv";
      break;
    case 2:
      return "_dth";
      break;
    case 3:
      return "_slhf";
      break;
    case 4:
      return "_sshf";
      break;
    case 5:
      return "_slhf";
      break;
    case 6:
      return "_tp";
      break;
    case 7:
      return "_IWV_";
      break;
    case 8:
      return "_CAO_";
      break;
    default: 
      return "_";
      break;
  }
}

function getProbKind()
{
  switch (kind) {
    case 1:
      return "_ar";
      break;
    case 2:
      return "_cao";
      break;
    case 3:
      return "_ehs";
      break;
    case 4:
      return "_sshf";
      break;
    case 5:
      return "_slhf";
      break;
    case 6:
      return "_tp";
      break;
    case 7:
      return "_IWV";
      break;
    case 8:
      return "_CAO";
      break;
    default: 
      return "_";
      break;
  }
}

function getDatename() 
{
	if ((cday.getUTCMonth()+1)<10) {
		mfill="0";
	} else {
		mfill="";
	}
	if (cday.getUTCDate()<10) {
		dfill="0";
	} else {
		dfill="";
	}
	return cday.getUTCFullYear()+mfill+(cday.getUTCMonth()+1)+dfill+cday.getUTCDate();
}

function getFcdate() 
{
	var dday=new Date(cday);
	dday.setUTCHours(dday.getUTCHours()+fdate);
	if ((dday.getUTCMonth()+1)<10) {
		mfill="0";
	} else {
		mfill="";
	}
	if (dday.getUTCDate()<10) {
		dfill="0";
	} else {
		dfill="";
	}
	if (dday.getUTCHours()<10) {
		hfill="0";
	} else {
		hfill="";
	}
	return dday.getUTCFullYear()+mfill+(dday.getUTCMonth()+1)+dfill+dday.getUTCDate()+"_"+hfill+dday.getUTCHours();
}

function getFcStep() 
{
	if (fdate<10) {
		mfill="0";
	} else {
		mfill="";
	}
	if (fdate<100) {
		dfill="0";
	} else {
		dfill="";
	}
	return "t+"+mfill+dfill+fdate;
}

function getMember(mem)
{
	mfill="";
	if (mem<10) {
	  mfill="0";
  }
	return mfill+mem;
}

function getBasetime()
{
  if (cday.getUTCHours() == 0 ) {
    return "00";
  } else {
    return "12";
  }
}

function getFilename(type)
{
        if (kind>=7) {
  	return "./gfx/fc_"+getDatename()+"/EPS_heatmap_"+getDatename()+getBasetime()+getKind()+thresholds[threshold]+".png";
	}

	dzoom="";
	if (zoom==true) {
	  dzoom="_zoom";
	}
	if (type==1) {
  	return "./gfx/fc_"+getDatename()+"/fc_"+getDatename()+"_"+getBasetime()+"_"+getFcStep()+"_"+thresholds[threshold]+getProbKind()+"_probability"+dzoom+".gif";
	} else if (type==2) {
  	return "./gfx/fc_"+getDatename()+"/fc_"+getDatename()+"_"+getBasetime()+"_"+getFcStep()+"_m"+getMember(member1)+getKind()+dzoom+".gif";
	} else if (type==3) {
  	return "./gfx/fc_"+getDatename()+"/fc_"+getDatename()+"_"+getBasetime()+"_"+getFcStep()+"_m"+getMember(member2)+getKind()+dzoom+".gif";
	}
	return "./gfx/fc_"+getDatename()+"/fc_"+getDatename()+"_"+getBasetime()+"_"+getFcStep()+getKind()+"_ensmean"+dzoom+".gif";
}

function prepareFigure() 
{
	if (kind>=7) {
	document.getElementById("plot_mean").src=getFilename(0);
	document.getElementById("plot_mean").alt=getFilename(0);
	//document.getElementById("plot_mean").width = "1000px";
	document.getElementById("plot_prob").style.display = "none";
	document.getElementById("plot_mem1").style.display = "none";
	document.getElementById("plot_mem2").style.display = "none";
	} else {
	document.getElementById("plot_mean").src=getFilename(0);
	document.getElementById("plot_mean").alt=getFilename(0);
	//document.getElementById("plot_mean").width = "500px";
	document.getElementById("plot_prob").style.display = "block";
	document.getElementById("plot_prob").src=getFilename(1);
	document.getElementById("plot_prob").alt=getFilename(1);
	document.getElementById("plot_mem1").style.display = "block";
	document.getElementById("plot_mem1").src=getFilename(2);
	document.getElementById("plot_mem1").alt=getFilename(2);
	document.getElementById("plot_mem2").style.display = "block";
	document.getElementById("plot_mem2").src=getFilename(3);
	document.getElementById("plot_mem2").alt=getFilename(3);
	}
}

function selectThreshold()
{
	threshold=document.getElementById("thresholds").selectedIndex;
	prepareFigure();
}

function checkZoom()
{
	zoom=document.selsize.zoom.checked;
	prepareFigure();
}

function initWebsite()
{
//if (cday.getUTCHours()<12) {
  	cday.setUTCHours(0,0,0);
//  	} else {
//  	cday.setUTCHours(12,0,0);
//  	}
  	member1=1;
	member2=1;
	period=0;
	site=0;
	type=0;
	threshold=0;
	selectThreshold();
	document.getElementById("btime").innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("ftime").innerHTML=getFcStep();
	document.getElementById("valid").innerHTML=getFcdate();
    document.getElementById("member1").innerHTML=getMember(member1);
    document.getElementById("member2").innerHTML=getMember(member2);
	prepareFigure();
}

function skiponeback() 
{
	cday.setUTCHours(cday.getUTCHours()-12);
	fdate+=12;
	if (fdate>240) {
		fdate=240;
	}
	document.getElementById("btime").innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("valid").innerHTML=getFcdate();
    document.getElementById("ftime").innerHTML=getFcStep();
	prepareFigure();
}

function skiponeforward() 
{
	cday.setUTCHours(cday.getUTCHours()+12);
	fdate-=12;
	if (fdate<0) {
		fdate=0;
	}
	document.getElementById("btime").innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("valid").innerHTML=getFcdate();
    document.getElementById("ftime").innerHTML=getFcStep();
	prepareFigure();
}

function skip6hforward() 
{
	fdate+=12;
	if (fdate>240) {
		fdate=240;
	}
	document.getElementById("btime").innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("ftime").innerHTML=getFcStep();
	document.getElementById("valid").innerHTML=getFcdate();
	prepareFigure();
}

function skip6hback() 
{
	fdate-=12;
	if (fdate<0) {
		fdate=0;
	}
	document.getElementById("btime").innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("ftime").innerHTML=getFcStep();
	document.getElementById("valid").innerHTML=getFcdate();
	prepareFigure();
}

function skip1mforward(m) 
{
  if (m==0) {
		member1+=1;
		if (member1>24) {
			member1=1;
		}
		document.getElementById("plot_mem1").src=getFilename(2);
		document.getElementById("plot_mem1").alt=getFilename(2);
		document.getElementById("member1").innerHTML=getMember(member1);
  } else {
		member2+=1;
		if (member2>24) {
			member2=1;
		}
		document.getElementById("plot_mem2").src=getFilename(3);
		document.getElementById("plot_mem2").alt=getFilename(3);
		document.getElementById("member2").innerHTML=getMember(member2);
  }
	prepareFigure();
}

function skip1mback(m) 
{
  if (m==0) {
		member1-=1;
		if (member1<1) {
			member1=24;
		}
		document.getElementById("plot_mem1").src=getFilename(2);
	  document.getElementById("plot_mem1").alt=getFilename(2);
		document.getElementById("member1").innerHTML=getMember(member1);
  } else {
		member2-=1;
		if (member2<1) {
			member2=24;
		}
		document.getElementById("plot_mem2").src=getFilename(3);
	  document.getElementById("plot_mem2").alt=getFilename(3);
		document.getElementById("member2").innerHTML=getMember(member2);
  }
	prepareFigure();
}

// fin
