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
var mkind=new Array(0,1);
var synced=true;
var kind = 1;
var tim = new Array(1,1);
var bt = 0;

// treshold settings and names
var varNames=new Array("CAO Index","OLR","Fluxes","d-excess","Z500, VEL, P","T850","Icing")
var captions=new Array(2);
captions[0]="";
captions[1]="";
captions[2]="";

// functions

function getChannel(m)
{
	switch (mkind[m]) {
		case 0:
                return "mne_channel5,2AA.png";
		break;
		case 1:
                return "mne_channel1,2,3AA.png";
		break;
		default:
		return "None";
		break;
	}
}

function getTime(n)
{
	var sel = document.getElementById("sel_time"+n);
        return sel.options[tim[n]].text;
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

function getFilename(k)
{
  return "./gfx/sat_"+getDatename()+"/noaa"+getDatename()+getTime(k)+getChannel(k);
}

function prepareFigure() 
{
	document.getElementById("panel1").src=getFilename(0);
	document.getElementById("panel1").alt=getFilename(0);
	document.getElementById("panel2").src=getFilename(1);
	document.getElementById("panel2").alt=getFilename(1);
}

function selectChannel(n)
{
    switch(n) {
 	case 0:
	  mkind[0]=document.getElementById("channel0").selectedIndex;
	  break;
 	case 1:
	  mkind[1]=document.getElementById("channel1").selectedIndex;
	  break;
    }
    prepareFigure();
}


function selectTime(n)
{
    switch(n) {
	case 0:
	  tim[0]=document.getElementById("sel_time0").selectedIndex;
	  document.getElementById("panel1").src=getFilename(0);
	  break;
	case 1:
	  tim[1]=document.getElementById("sel_time1").selectedIndex;
	  document.getElementById("panel2").src=getFilename(1);
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
	prepareFigure(); // prepare both rows
}

function skiponeback() 
{
	row=0;
	cday[row].setUTCHours(cday[row].getUTCHours()-24);
	fdate[row]+=24;
	document.getElementById("btime"+row).innerHTML=getDatename()+"_"+getBasetime();
	prepareFigure();
}

function skiponeforward() 
{
	row=0;
	cday[row].setUTCHours(cday[row].getUTCHours()+24);
	fdate[row]-=24;
	document.getElementById("btime"+row).innerHTML=getDatename()+"_"+getBasetime();
	prepareFigure();
}

// fin
