// javascript logic for the watersip website
// HS, 16.09.2016

window.onload=initWebsite;

// date settings
//var cday = new Date(2018,2,18);
var cday = new Date();
var fdate = 0; // forecast time step
var kind = 1;
var bt = 0;

// treshold settings and names
var thresholds=new Array("05mm","10mm","12mm");
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
      return "_IWV_";
      break;
    case 2:
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
      return "_IWV";
      break;
    case 2:
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
  	return "./gfx/fc_"+getDatename()+"/EPS_heatmap_"+getDatename()+getBasetime()+getKind()+thresholds[threshold]+".png";
}

function prepareFigure() 
{
	document.getElementById("plot_mean").src=getFilename(0);
	document.getElementById("plot_mean").alt=getFilename(0);
}

function selectThreshold()
{
	threshold=document.getElementById("thresholds").selectedIndex;
	prepareFigure();
}

function initWebsite()
{
//if (cday.getUTCHours()<12) {
  	cday.setUTCHours(0,0,0);
//  	} else {
//  	cday.setUTCHours(12,0,0);
//  	}
	period=0;
	site=0;
	type=0;
	threshold=0;
	selectThreshold();
	document.getElementById("btime").innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("ftime").innerHTML=getFcStep();
	document.getElementById("valid").innerHTML=getFcdate();
	prepareFigure();
}

function skiponeback() 
{
	cday.setUTCHours(cday.getUTCHours()-12);
	fdate=240;
	document.getElementById("btime").innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("valid").innerHTML=getFcdate();
    document.getElementById("ftime").innerHTML=getFcStep();
	prepareFigure();
}

function skiponeforward() 
{
	cday.setUTCHours(cday.getUTCHours()+12);
	fdate=240;
	document.getElementById("btime").innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("valid").innerHTML=getFcdate();
        document.getElementById("ftime").innerHTML=getFcStep();
	prepareFigure();
}

function skip6hforward() 
{
        fdate=240;
	document.getElementById("btime").innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("ftime").innerHTML=getFcStep();
	document.getElementById("valid").innerHTML=getFcdate();
	prepareFigure();
}

function skip6hback() 
{
	fdate=240;
	document.getElementById("btime").innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("ftime").innerHTML=getFcStep();
	document.getElementById("valid").innerHTML=getFcdate();
	prepareFigure();
}

function setKind(id) 
{
   document.getElementById('ARHM').bgColor="#a0a0a0";
   document.getElementById('CAOHM').bgColor="#a0a0a0";
   document.getElementById(id).bgColor="#aaccff";
   switch (id) {
     case 'ARHM':
       document.getElementById('ti_mean').innerHTML="Heat map of AR probability (0-100%) for 10-day forecast period";
       document.getElementById('th0').innerHTML="5 mm";
       document.getElementById('th1').innerHTML="10 mm";
       document.getElementById('th2').innerHTML="12 mm";
       thresholds[0]="05mm";
       thresholds[1]="10mm";
       thresholds[2]="12mm";
       kind=1;
       break;
     case 'CAOHM':
       document.getElementById('ti_mean').innerHTML="Heat map of CAO probability (0-100%) for 10-day forecast period";
       document.getElementById('th0').innerHTML="2 K";
       document.getElementById('th1').innerHTML="4 K";
       document.getElementById('th2').innerHTML="8 K";
       thresholds[0]="02K";
       thresholds[1]="04K";
       thresholds[2]="08K";
       kind=2;
       break;
     default:
       // do nothing
       break;
   }
   prepareFigure();
   return false;
}

// fin
