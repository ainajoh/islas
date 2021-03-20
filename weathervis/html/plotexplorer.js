// logic for plotexplorer
//-------------------------------------------
// Harald Sodemann, 24.11.2019
//-------------------------------------------

// new function on Date class to add days
Date.prototype.addDays = function(days) {
    var date = new Date(this.valueOf());
    date.setDate(date.getDate() + days);
    return date;
}

// Get the ISO week date week number
Date.prototype.getWeek = function () {
  var target = new Date(this.valueOf());
  var dayNr = (this.getDay() + 6) % 7;
  target.setDate(target.getDate() - dayNr + 3);
  var firstThursday = target.valueOf();
  target.setMonth(0, 1);
  if (target.getDay() !== 4) {
    target.setMonth(0, 1 + ((4 - target.getDay()) + 7) % 7);
  }
  return 1 + Math.ceil((firstThursday - target) / 604800000);
}

// names and titles for deployments, platforms and variables
var dnam = ["ISLAS2021",];
var dtit = ["ISLAS2021 at Andoya",];
var dsta = ['2021-03-16'];
// find start and end periods for each campaign
for (i=0; i<dsta.length; i++) {
    aday = new Date(dsta[i]);
    dsta[i]=currentPeriod(aday);
}
var dend = ['2021-04-01'];
for (i=0; i<dend.length; i++) {
    aday = new Date(dend[i]);
    dend[i]=currentPeriod(aday.addDays(7));
}
var pnam = ["MRR","DSD","CRDS","WSIP",];
var ptit = ["Micro rain radar","Disdrometer","Picarro L2130-i (HIDS2254)","Watersip",];
var vnam = ["Z","RR","W","LWC","Dm","RR","SND","logND","VIS","CNT","REFL","T","q","dD","d18O","dxs","SLAT","SLON","LFR","TTIM","SSKT","TRD","DXS","CT",];
var vtit = ["Reflectivity","Precipitation rate","Vertical velocity","Liquid water content","Mean diameter","Precipitation rate","Size spectrum","Size classes","Visibility","Droplet count","Reflectivity","Housing temperature","Specific humidity","delta D","delta O-18","d-excess","Source latitude","Source longitude","Land fraction","Transport time","Source skin temperature","Transport distance","Deuterium excess","Condensation temperature",];
var vuni = ["dBz","mm h-1","m s-1","g kg-1","mm","mm h-1","1","1","m","1","dBz","C","g kg-1","permil","permil","permil","deg N","deg E","percent","hours","deg C","km","permil","deg C",];
var caps = ["Reflectivity","Precipitation rate from bin 2","Vertical fall speed","Liquid water content in air","Mean diameter","Precipitation rate","Size spectrum","Size classes","Visibility","Droplet count","Reflectivity","Housing temperature","Specific humidity from Picarro L2130-i","D isotope ratio in vapour from Picarro L2130-i","O-18 isotope ratio in vapour from Picarro L2130-i","deuterium excess in vapour from Picarro L2130-i","Moisture source mean latitude (vapour analysis)","Moisture source mean longitude (vapour analysis)","Moisture source mean land fraction (vapour analysis)","Moisture source mean transport time (vapour analysis)","Moisture source mean source SKT (vapour analysis)","Moisture source mean transport distance (vapour analysis)","Moisture source mean deuterium excess (vapour analysis)","Moisture source mean condensation temperature (vapour analysis)",]
// length and indices for platforms
var plen = [4];
var pstart = [0];
for (i=0; i<plen.length; i++) {
    pstart.push(pstart[i]+plen[i]);
}
var vlen = [4, 8, 4, 8];
var vstart = [0];
for (i=0; i<vlen.length; i++) {
    vstart.push(vstart[i]+vlen[i]);
}

// current indices
var insind = 0;
var parind = 0;
var cday = currentPeriod(new Date());
var eday = cday.addDays(7);
var mnam = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
// set the current period

// return the current Friday to Friday period
function currentPeriod(date)
{
    weekday=date.getDay();
    if (weekday<5) {
        date=date.addDays(-7+(5-weekday));
    }
    if (weekday==6) {
        date=date.addDays(-1);
    }
    return date;
}

function setInitialPeriod()
{
  document.getElementById("Index").innerHTML="Week "+pad(cday.getWeek())+" ("+fmtDateWeek(cday," ")+" to "+fmtDateWeek(eday," ")+" "+eday.getFullYear()+")"
}

function pad(nr)
{
	if (nr<10) {
		return "0"+nr;
	}
	return nr;
}

function fmtDate(d,sep) {
	return [  d.getFullYear(),
	  ('0' + (d.getMonth() + 1)).slice(-2),
	  ('0' + d.getDate()).slice(-2) ].join(sep);
}

function fmtDateWeek(d,sep) {
	return [ ('0' + d.getDate()).slice(-2),
	  mnam[d.getMonth() + 0] ].join(sep);
}

// advance to next week
function nextWeek()
{
	// determine date range
	cday=cday.addDays(7);
	eday=eday.addDays(7);
	document.getElementById("Index").innerHTML="Week "+pad(cday.getWeek())+" ("+fmtDateWeek(cday," ")+" to "+fmtDateWeek(eday," ")+" "+eday.getFullYear()+")"
	// update image
  refreshImage(0);
  refreshImage(1);
}

// return to previous week
function prevWeek()
{
	// determine date range
	cday=cday.addDays(-7);
	eday=eday.addDays(-7);
	document.getElementById("Index").innerHTML="Week "+pad(cday.getWeek())+" ("+fmtDateWeek(cday," ")+" to "+fmtDateWeek(eday," ")+" "+eday.getFullYear()+")"
	// update image
    refreshImage(0);
    refreshImage(1);
}

// advance to next month
function nextMonth()
{
	// determine date range
	cday=cday.addDays(28);
	eday=eday.addDays(28);
	document.getElementById("Index").innerHTML="Week "+pad(cday.getWeek())+" ("+fmtDateWeek(cday," ")+" to "+fmtDateWeek(eday," ")+" "+eday.getFullYear()+")"
	// update image
    refreshImage(0);
    refreshImage(1);
}

// return to previous month
function prevMonth()
{
	// determine date range
	cday=cday.addDays(-28);
	eday=eday.addDays(-28);
	document.getElementById("Index").innerHTML="Week "+pad(cday.getWeek())+" ("+fmtDateWeek(cday," ")+" to "+fmtDateWeek(eday," ")+" "+eday.getFullYear()+")"
	// update image
  refreshImage(0);
  refreshImage(1);
}

// create innerHTML with options for current deployment
function writeInstruments()
{
    depind=document.getElementById("deployment").selectedIndex;
    sidx=pstart[depind];
    eidx=pstart[depind+1];
    ret="";
    for (i=sidx; i<eidx; i++) {
        ret+="<option>"+ptit[i]+"</option>";
    }
    return ret;
}

// create innerHTML with options for platform n
function writeVariables(n)
{
    depind=document.getElementById("deployment").selectedIndex;
    insind=document.getElementById("instrument"+n).selectedIndex;
    sidx=vstart[pstart[depind]+insind];
    eidx=vstart[pstart[depind]+insind+1];
    ret="";
    for (i=sidx; i<eidx; i++) {
        ret+="<option>"+vnam[i]+" ("+vtit[i]+")</option>";
    }
    return ret;
}

// react to selection of a deployment
function selDeployment(n)
{
  depind=document.getElementById("deployment").selectedIndex;
  // change parameters depending on instrument
  document.getElementById("instrument0").innerHTML=writeInstruments();
  document.getElementById("instrument1").innerHTML=writeInstruments();
  selInstrument(0);
  selInstrument(1);
}

// react to selection of an instrument
function selInstrument(n)
{
  insind=document.getElementById("instrument"+n).selectedIndex;
  // change parameters depending on instrument
  document.getElementById("parameter"+n).innerHTML=writeVariables(n);
  refreshImage(n);
}

// react to selection of a parameter
function selParameter(n)
{
	refreshImage(n);
}

// transform current choice in image file name
function refreshImage(n)
{
  // transform into filename part
  dat1=fmtDate(cday,"");
  dat2=fmtDate(eday,"");

  depind=document.getElementById("deployment").selectedIndex;
  insind=document.getElementById("instrument"+n).selectedIndex;
  parind=document.getElementById("parameter"+n).selectedIndex;
  plt=pnam[pstart[depind]+insind];
  par=vnam[vstart[pstart[depind]+insind]+parind];
  // set image source
  document.images[n].src = "gfx/"+dnam[depind]+"/"+plt+"_"+par+"_"+dat1+"-"+dat2+".png";
  // set caption
  document.getElementById("caption"+n).innerHTML = caps[vstart[pstart[depind]+insind]+parind]+" ("+vuni[vstart[pstart[depind]+insind]+parind]+")";
}
