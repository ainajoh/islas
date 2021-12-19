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
var synced=true;
var kind = 1;
var domain = new Array(2,4);
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

function getLevel(n)
{
  switch (n) {
    case 0:
      return "_L00300";
      break;
    case 1:
      return "_L01000";
      break;
    case 2:
      return "_L01500";
      break;
    case 3:
      return "_L02000";
      break;
    case 4:
      return "_L02500";
      break;
    case 5:
      return "_L03000";
      break;
    case 6:
      return "_L00000";
      break;
    default: 
      return "_";
      break;
  }
}

function getDomainname(n)
{
	switch (domain[n]) {
		case 0:
		return "NorwegianSea_area";
		break;
		case 1:
		return "Andenes_area";
		break;
		case 2:
		return "Svalbard";
		break;
		case 3:
		return "North_Norway";
		break;
		case 4:
		return "AromeArctic";
		break;
		case 5:
		return "South_Norway";
		break;
		case 6:
		return "West_Norway";
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
  return "./gfx/"+getDatename(k)+getBasetime(k)+"/FLEXPART_AA_"+getDomainname(k)+getLevel(n)+"_"+getDatename(k)+getBasetime(k)+getFcStep(k)+".png";
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
    	cday[0].setUTCHours(0,0,0);
    	}
        if (cday[1].getUTCHours()<12) {
  	cday[1].setUTCHours(0,0,0);
    	} else {
    	cday[1].setUTCHours(0,0,0);
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
	if ((fdate[row]>24) && (fdate[row]<36)) {
		fdate[row]=fdate[row] - (fdate[row] % 3)
	}
	if ((fdate[row]>36) && (fdate[row]<66)) {
		fdate[row]=fdate[row] - (fdate[row] % 6)
	}
	if (fdate[row]>66) {
		fdate[row]=66;
	}
	document.getElementById("btime"+row).innerHTML=getDatename(row)+"_"+getBasetime(row);
	document.getElementById("valid"+row).innerHTML=getFcdate(row);
	document.getElementById("ftime"+row).innerHTML=getFcStep(row);
	prepareFigure(row);
	syncFigures(row);
}

function skiponeforward(row) 
{
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
	document.getElementById("btime"+row).innerHTML=getDatename(row)+"_"+getBasetime(row);
	document.getElementById("valid"+row).innerHTML=getFcdate(row);
	document.getElementById("ftime"+row).innerHTML=getFcStep(row);
	prepareFigure(row);
	syncFigures(row);
}

function skip1hforward(row) 
{
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
	document.getElementById("btime"+row).innerHTML=getDatename(row)+"_"+getBasetime(row);
	document.getElementById("valid"+row).innerHTML=getFcdate(row);
	document.getElementById("ftime"+row).innerHTML=getFcStep(row);
	prepareFigure(row);
	syncFigures(row);
}

function skip1hback(row) 
{
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
	document.getElementById("btime"+row).innerHTML=getDatename(row)+"_"+getBasetime(row);
	document.getElementById("valid"+row).innerHTML=getFcdate(row);
	document.getElementById("ftime"+row).innerHTML=getFcStep(row);
	prepareFigure(row);
	syncFigures(row);
}

function syncFigures(row)
{
	if (synced==true) {
		if (row<1) {
			cday[1]=cday[0];
			fdate[1]=fdate[0];
			document.getElementById("btime"+1).innerHTML=getDatename(1)+"_"+getBasetime(1);
			document.getElementById("valid"+1).innerHTML=getFcdate(1);
			document.getElementById("ftime"+1).innerHTML=getFcStep(1);
			prepareFigure(1);
		} else {
			cday[0]=cday[1];
			fdate[0]=fdate[1];
			document.getElementById("btime"+0).innerHTML=getDatename(0)+"_"+getBasetime(0);
			document.getElementById("valid"+0).innerHTML=getFcdate(0);
			document.getElementById("ftime"+0).innerHTML=getFcStep(0);
			prepareFigure(0);
		}
	}
}


// fin
