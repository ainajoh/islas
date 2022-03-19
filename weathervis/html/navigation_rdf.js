// javascript logic for the watersip website
// HS, 16.09.2016

window.onload=initWebsite;

// date settings
var day0 = new Date(Date.now());
hrs=day0.getUTCHours();
if (hrs<14) {
  day0.setUTCHours(0);
  dy=day0.getDate();
  day0.setDate(dy-1);
}
day0.setUTCMinutes(0);
day0.setUTCSeconds(0);
var day1 = new Date(Date.now());
day1.setDate(day0.getDate());
day1.setUTCHours(day0.getUTCHours());
day1.setUTCMinutes(0);
day1.setUTCSeconds(0);
var cday = new Array(day0, day1);
var fdate = new Array(0,0); // forecast time step
var vkind=new Array(0,0);
var mkind=new Array(0,0);
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

function getVariable(m)
{
	switch (mkind[m]) {
		case 0:
		return "p";
		break;
		case 1:
		return "lon";
		break;
		case 2:
		return "lat";
		break;
		case 3:
		return "alt";
		break;
		case 4:
		return "vel";
		break;
		case 5:
		return "w";
		break;
		case 6:
		return "hdist";
		break;
		case 7:
		return "hsep";
		break;
		case 8:
		return "vdist";
		break;
		case 9:
		return "vsep";
		break;
		case 10:
		return "bldist";
		break;
		case 11:
		return "tdist";
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
  return "./gfx/fc_"+getDatename(k)+"/RDF_map_"+getDatename(k)+getBasetime(k)+getFcStep()+'_36-48_'+getVariable(n)+".png";
}

function prepareFigure() 
{
	document.getElementById("panel1").src=getFilename(vkind[0],0);
	document.getElementById("panel1").alt=getFilename(vkind[0],0);
	document.getElementById("panel2").src=getFilename(vkind[1],1);
	document.getElementById("panel2").alt=getFilename(vkind[1],1);
}

function selectVariable(n)
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
	document.getElementById("ftime"+row).innerHTML=getFcStep();
	prepareFigure(); // prepare both rows
}

function skiponeback() 
{
	row=0;
	cday[row].setUTCHours(cday[row].getUTCHours()-24);
	fdate[row]+=24;
	if (fdate[row]>60) {
		fdate[row]=60;
	}
	document.getElementById("btime"+row).innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("valid"+row).innerHTML=getFcdate();
	document.getElementById("ftime"+row).innerHTML=getFcStep();
	prepareFigure();
}

function skiponeforward() 
{
	row=0;
	cday[row].setUTCHours(cday[row].getUTCHours()+24);
	fdate[row]-=24;
	if (fdate[row]<0) {
		fdate[row]=0;
	}
	document.getElementById("btime"+row).innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("valid"+row).innerHTML=getFcdate();
	document.getElementById("ftime"+row).innerHTML=getFcStep();
	prepareFigure();
}

function skip1hforward() 
{
	row=0;
	fdate[row]+=12;
	if (fdate[row]>60) {
		fdate[row]=60;
	}
	document.getElementById("btime"+row).innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("valid"+row).innerHTML=getFcdate();
	document.getElementById("ftime"+row).innerHTML=getFcStep();
	prepareFigure();
}

function skip1hback() 
{
	row=0;
	fdate[row]-=12;
	if (fdate[row]<0) {
		fdate[row]=0;
	}
	document.getElementById("btime"+row).innerHTML=getDatename()+"_"+getBasetime();
	document.getElementById("valid"+row).innerHTML=getFcdate();
	document.getElementById("ftime"+row).innerHTML=getFcStep();
	prepareFigure();
}

// fin
