// javascript logic for the watersip website
// HS, 16.09.2016

window.onload=initWebsite;

// date settings
var day0 = new Date(Date.now());
day0.setUTCHours(0);
day0.setUTCMinutes(0);
day0.setUTCSeconds(0);
var day1 = new Date(Date.now());
day1.setUTCHours(0);
day1.setUTCMinutes(0);
day1.setUTCSeconds(0);
var cday = new Array(day0, day1);
var fdate = new Array(0,0); // forecast time step
var vkind=new Array(4,4);
var mkind=new Array(0,1);
var synced=true;
var kind = 1;
var domain = new Array(0,0);
var bt = 0;

// treshold settings and names
var varNames=new Array("CAO Index","OLR","Fluxes","d-excess","Z500, VEL, P","T850","Icing")
var captions=new Array(2);
captions[0]="";
captions[1]="";
captions[2]="";

// functions
function checkSync()
{
  synced=document.getElementById("sync").checked;
}

function getKind(n)
{
  switch (n) {
    case 4:
      return "";
      break;
    case 0:
      return "_[SEA]";
      break;
    case 1:
      return "_[LAND]";
      break;
    case 2:
      return "_[ALL_DOMAIN]";
      break;
    case 3:
      return "_[ALL_NEAREST]";
      break;
    default: 
      return "_";
      break;
  }
}

function getMeteogram(m)
{
	switch (mkind[m]) {
		case 0:
		return "op1";
		break;
		case 1:
		return "op2";
		break;
		case 2:
		break;
		default:
		return "None";
		break;
	}
}


function getDomainname(n)
{
	switch (domain[n]) {
		case 0:
		return "VPMETEOGRAM_Andenes_";
		break;
		case 1:
		return "VPMETEOGRAM_ALOMAR_";
		break;
		case 2:
		return "VPMETEOGRAM_Tromso_";
		break;
		case 3:
		return "VPMETEOGRAM_NyAlesund_";
		break;
		case 4:
		return "VPMETEOGRAM_Longyearbyen_";
		break;
		case 5:
		return "VPMETEOGRAM_Bjornoya_";
		break;
		case 6:
		return "VPMETEOGRAM_NorwegianSea_";
		break;
		case 7:
		return "VPMETEOGRAM_CAO_";
		break;
	        case 8:
                return "VPMETEOGRAM_pcmet1_";
                break;
	        case 9:
                return "VPMETEOGRAM_pcmet2_";
                break;
		case 10:
                return "VPMETEOGRAM_pcmet3_";
                break;
		default:
		return "None";
		break;
	}
}

function getDatename() 
{
	row=0;
	if ((cday[row].getUTCMonth()+1)<10) {
		mfill="0";
	} else {
		mfill="";
	}
	if (cday[row].getUTCDate()<10) {
		dfill="0";
	} else {
		dfill="";
	}
	return cday[row].getUTCFullYear()+mfill+(cday[row].getUTCMonth()+1)+dfill+cday[row].getUTCDate();
}

function getFcdate() 
{
	row=0;
	var dday=new Date(cday[row]);
	dday.setUTCHours(dday.getUTCHours()+fdate[row]);
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
	row=0;
	if (fdate[row]<10) {
		mfill="0";
	} else {
		mfill="";
	}
	return "+"+mfill+fdate[row];
}

function getBasetime()
{
	row=0;
	bfill="";
	btim=cday[row].getUTCHours();
	if (btim<10) {
	  bfill="0";
  	}
	return bfill+btim;
}

function getFilename(k,n)
{
  return "./gfx/"+getDatename(k)+getBasetime(k)+"/"+getDomainname(n)+getDatename(k)+getBasetime(k)+"_"+getMeteogram(n)+".png";
}

function prepareFigure() 
{
	document.getElementById("panel1").src=getFilename(vkind[0],0);
	document.getElementById("panel1").alt=getFilename(vkind[0],0);
	document.getElementById("panel2").src=getFilename(vkind[1],1);
	document.getElementById("panel2").alt=getFilename(vkind[1],1);
}

function selectMeteogram(n)
{
    switch(n) {
 	case 0:
	  mkind[0]=document.getElementById("meteogram1").selectedIndex;
	  break;
 	case 1:
	  mkind[1]=document.getElementById("meteogram2").selectedIndex;
	  break;
    }
    prepareFigure();
}


function selectPoint(n)
{
    switch(n) {
 	case 0:
	  vkind[0]=document.getElementById("domain1").selectedIndex;
	  break;
 	case 1:
	  vkind[1]=document.getElementById("domain2").selectedIndex;
	  break;
    }
    prepareFigure();
}

function selectLocation(n)
{
    switch(n) {
	case 0:
	  domain[0]=document.getElementById("sel_v1").selectedIndex;
	  document.getElementById("panel1").src=getFilename(vkind[0],0);
	  break;
	case 1:
	  domain[1]=document.getElementById("sel_v2").selectedIndex;
	  document.getElementById("panel2").src=getFilename(vkind[1],1);
	  break;
        default:
	  break;
	}
  prepareFigure();
}

function initWebsite()
{
        if (cday[0].getUTCHours()<12) {
  	cday[0].setUTCHours(0,0,0);
    	} else {
    	cday[0].setUTCHours(12,0,0);
    	}
        if (cday[1].getUTCHours()<12) {
  	cday[1].setUTCHours(0,0,0);
    	} else {
    	cday[1].setUTCHours(12,0,0);
    	}
	period=0;
	site=0;
	type=0;
        row=0
	document.getElementById("btime"+row).innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("valid"+row).innerHTML=getFcdate();
	//document.getElementById("ftime"+row).innerHTML=getFcStep();
	prepareFigure(); // prepare both rows
}

function skiponeback() 
{
	row=0;
	cday[row].setUTCHours(cday[row].getUTCHours()-6);
	fdate[row]+=6;
	if ((fdate[row]>24) && (fdate[row]<36)) {
		fdate[row]=fdate[row] - (fdate[row] % 3)
	}
	if ((fdate[row]>36) && (fdate[row]<66)) {
		fdate[row]=fdate[row] - (fdate[row] % 6)
	}
	if (fdate[row]>66) {
		fdate[row]=66;
	}
	document.getElementById("btime"+row).innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("valid"+row).innerHTML=getFcdate();
	//document.getElementById("ftime"+row).innerHTML=getFcStep();
	prepareFigure();
}

function skiponeforward() 
{
	row=0;
	cday[row].setUTCHours(cday[row].getUTCHours()+6);
	fdate[row]-=6;
	if ((fdate[row]>24) && (fdate[row]<36)) {
		fdate[row]=fdate[row] + (fdate[row] % 3)
	}
	if ((fdate[row]>36) && (fdate[row]<66)) {
		fdate[row]=fdate[row] + (fdate[row] % 6)
	}
	if (fdate[row]<0) {
		fdate[row]=0;
	}
	document.getElementById("btime"+row).innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("valid"+row).innerHTML=getFcdate();
	//document.getElementById("ftime"+row).innerHTML=getFcStep();
	prepareFigure();
}

function skip1hforward() 
{
	row=0;
	fdate[row]+=1;
	if (fdate[row]>24) {
		fdate[row]=fdate[row]+2;
	}
	if (fdate[row]>36) {
		fdate[row]=fdate[row]+3;
	}
	if (fdate[row]>66) {
		fdate[row]=66;
	}
	document.getElementById("btime"+row).innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("valid"+row).innerHTML=getFcdate();
	//document.getElementById("ftime"+row).innerHTML=getFcStep();
	prepareFigure();
}

function skip1hback() 
{
	row=0;
	if (fdate[row]>36) {
		fdate[row]=fdate[row]-3;
	}
	if (fdate[row]>24) {
		fdate[row]=fdate[row]-2;
	}
	fdate[row]-=1;
	if (fdate[row]<0) {
		fdate[row]=0;
	}
	document.getElementById("btime"+row).innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("valid"+row).innerHTML=getFcdate();
	//document.getElementById("ftime"+row).innerHTML=getFcStep();
	prepareFigure();
}

// fin
