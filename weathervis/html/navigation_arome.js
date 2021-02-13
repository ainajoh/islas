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
var vkind=new Array(0,1,2,3,4,5);
var synced=false;
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
  synced=false
}

function getKind(n)
{
  switch (n) {
    case 0:
      return "_CAOi";
      break;
    case 1:
      return "_OLR_sat";
      break;
    case 2:
      return "_surf";
      break;
    case 3:
      return "_Z500_VEL_P";
      break;
    case 4:
      return "_T850_RH";
      break;
    case 5:
      return "_dxs";
      break;
    case 6:
      return "_BLH";
      break;
    case 7:
      return "_clouds";
      break;
    default: 
      return "_";
      break;
  }
}

function getDomainname(n)
{
k=0;
if (n>2) {
	k=1;
}
	switch (domain[k]) {
		case 0:
		return "AromeArctic_AromeArctic";
		break;
		case 1:
		return "MEPS_MEPS";
		break;
		case 2:
		return "AromeArctic_Svalbard";
		break;
		case 3:
		return "AromeArctic_North_Norway";
		break;
		default:
		return "None";
		break;
	}
}

function getDatename(row) 
{
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

function getFcdate(row) 
{
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

function getFcStep(row) 
{
	if (fdate[row]<10) {
		mfill="0";
	} else {
		mfill="";
	}
	return "+"+mfill+fdate[row];
}

function getBasetime(row)
{
	bfill="";
	btim=cday[row].getUTCHours();
	if (btim<10) {
	  bfill="0";
  	}
	return bfill+btim;
}

function getFilename(n,k)
{
  return "./gfx/"+getDatename(k)+getBasetime(k)+"/"+getDomainname(k)+getKind(n)+"_"+getDatename(k)+getBasetime(k)+getFcStep(k)+".png";
}

function prepareFigure(n) 
{
if (n==0 || n>1) {
	document.getElementById("panel1").src=getFilename(vkind[0],0);
	document.getElementById("panel1").alt=getFilename(vkind[0],0);
	document.getElementById("panel2").src=getFilename(vkind[1],0);
	document.getElementById("panel2").alt=getFilename(vkind[1],0);
	document.getElementById("panel3").src=getFilename(vkind[2],0);
	document.getElementById("panel3").alt=getFilename(vkind[2],0);
}
if (n>=1) {
	document.getElementById("panel4").src=getFilename(vkind[3],1);
	document.getElementById("panel4").alt=getFilename(vkind[3],1);
	document.getElementById("panel5").src=getFilename(vkind[4],1);
	document.getElementById("panel5").alt=getFilename(vkind[4],1);
	document.getElementById("panel6").src=getFilename(vkind[5],1);
	document.getElementById("panel6").alt=getFilename(vkind[5],1);
}
}

function selectDomain(n)
{
    switch(n) {
 	case 0:
	domain[0]=document.getElementById("domain1").selectedIndex;
	break;
 	case 1:
	domain[1]=document.getElementById("domain2").selectedIndex;
	break;
    }
    prepareFigure(2);
}

function selectVar(n)
{
    switch(n) {
	case 1:
	  vkind[0]=document.getElementById("sel_v1").selectedIndex;
	  document.getElementById("panel1").src=getFilename(vkind[0],0);
	  break;
	case 2:
	  vkind[1]=document.getElementById("sel_v2").selectedIndex;
	  document.getElementById("panel2").src=getFilename(vkind[1],0);
	  break;
	case 3:
	  vkind[2]=document.getElementById("sel_v3").selectedIndex;
	  document.getElementById("panel3").src=getFilename(vkind[2],0);
	  break;
	case 4:
	  vkind[3]=document.getElementById("sel_v4").selectedIndex;
	  document.getElementById("panel4").src=getFilename(vkind[3],1);
	  break;
	case 5:
	  vkind[4]=document.getElementById("sel_v5").selectedIndex;
	  document.getElementById("panel5").src=getFilename(vkind[4],1);
	  break;
	case 6:
	  vkind[5]=document.getElementById("sel_v6").selectedIndex;
	  document.getElementById("panel6").src=getFilename(vkind[5],1);
	  break;
        default:
	  break;
	}
  prepareFigure(2);
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
	document.getElementById("btime"+row).innerHTML=getDatename(row)+"_"+getBasetime(row);
	document.getElementById("valid"+row).innerHTML=getFcdate(row);
	document.getElementById("ftime"+row).innerHTML=getFcStep(row);
        row=1
	document.getElementById("btime"+row).innerHTML=getDatename(row)+"_"+getBasetime(row);
	document.getElementById("valid"+row).innerHTML=getFcdate(row);
	document.getElementById("ftime"+row).innerHTML=getFcStep(row);
	prepareFigure(2); // prepare both rows
}

function skiponeback(row) 
{
	cday[row].setUTCHours(cday[row].getUTCHours()-6);
	fdate[row]+=6;
	if (fdate[row]>66) {
		fdate[row]=66;
	}
	document.getElementById("btime"+row).innerHTML=getDatename(row)+"_"+getBasetime(row);
	document.getElementById("valid"+row).innerHTML=getFcdate(row);
	document.getElementById("ftime"+row).innerHTML=getFcStep(row);
	prepareFigure(row);
}

function skiponeforward(row) 
{
	cday[row].setUTCHours(cday[row].getUTCHours()+6);
	fdate[row]-=6;
	if (fdate[row]<0) {
		fdate[row]=0;
	}
	document.getElementById("btime"+row).innerHTML=getDatename(row)+"_"+getBasetime(row);
	document.getElementById("valid"+row).innerHTML=getFcdate(row);
	document.getElementById("ftime"+row).innerHTML=getFcStep(row);
	prepareFigure(row);
}

function skip1hforward(row) 
{
	fdate[row]+=1;
	if (fdate[row]>66) {
		fdate[row]=66;
	}
	document.getElementById("btime"+row).innerHTML=getDatename(row)+"_"+getBasetime(row);
	document.getElementById("valid"+row).innerHTML=getFcdate(row);
	document.getElementById("ftime"+row).innerHTML=getFcStep(row);
	prepareFigure(row);
}

function skip1hback(row) 
{
	fdate[row]-=1;
	if (fdate[row]<0) {
		fdate[row]=0;
	}
	document.getElementById("btime"+row).innerHTML=getDatename(row)+"_"+getBasetime(row);
	document.getElementById("valid"+row).innerHTML=getFcdate(row);
	document.getElementById("ftime"+row).innerHTML=getFcStep(row);
	prepareFigure(row);
}


// fin
