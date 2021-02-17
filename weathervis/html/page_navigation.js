// page_navigation.js
// ----------------------
// HS, 15.08.2017
// ----------------------

function setKind(id) 
{
   document.getElementById('ARID').bgColor="#c0c0c0";
   document.getElementById('CAOID').bgColor="#c0c0c0";
   document.getElementById('EHSID').bgColor="#c0c0c0";
   document.getElementById('TPID').bgColor="#c0c0c0";
   document.getElementById('SSHID').bgColor="#c0c0c0";
   document.getElementById('SLOEID').bgColor="#c0c0c0";
   document.getElementById(id).bgColor="#aaccff";
   switch (id) {
     case 'ARID':
       document.getElementById('ti_memb1').innerHTML="member IWV (mm)";
       document.getElementById('ti_memb2').innerHTML="member IWV (mm)";
       document.getElementById('ti_mean').innerHTML="50-member ensemble mean IWV (mm)";
       document.getElementById('ti_prob').innerHTML="AR probability according to threshold (1/0)";
       thresholds[0]="th14"
       thresholds[1]="th20"
       thresholds[2]="th20"
       document.getElementById('th0').innerHTML="14mm";
       document.getElementById('th1').innerHTML="20mm";
       document.getElementById('th2').innerHTML="20mm";
       kind=1;
       break;
     case 'CAOID':
       document.getElementById('ti_memb1').innerHTML="member delta theta (K)";
       document.getElementById('ti_memb2').innerHTML="member delta theta (K)";
       document.getElementById('ti_mean').innerHTML="50-member ensemble mean delta theta (K)";
       document.getElementById('ti_prob').innerHTML="CAO probability according to threshold (1/0)";
       document.getElementById('th0').innerHTML="2 K";
       document.getElementById('th1').innerHTML="4 K";
       document.getElementById('th2').innerHTML="8 K";
       thresholds[0]="th02K";
       thresholds[1]="th04K";
       thresholds[2]="th08K";
       kind=2;
       break;
     case 'EHSID':
       document.getElementById('ti_memb1').innerHTML="member latent heat flux (W m-2)";
       document.getElementById('ti_memb2').innerHTML="member latent heat flux (W m-2)";
       document.getElementById('ti_mean').innerHTML="50-member ensemble mean latent heat flux (W m-2)";
       document.getElementById('ti_prob').innerHTML="EHS probability according to threshold (1/0)";
       document.getElementById('th0').innerHTML="100 W m-2";
       document.getElementById('th1').innerHTML="200 W m-2";
       document.getElementById('th2').innerHTML="200 W m-2";
       thresholds[0]="th0100";
       thresholds[1]="th0200";
       thresholds[2]="th0200";
       kind=3;
       break;
     case 'SSHID':
       document.getElementById('ti_memb1').innerHTML="member sensible heat flux (W m-2)";
       document.getElementById('ti_memb2').innerHTML="member sensible heat flux (W m-2)";
       document.getElementById('ti_mean').innerHTML="50-member ensemble mean sensible heat flux (W m-2)";
       document.getElementById('ti_prob').innerHTML="SSH probability according to threshold (1/0)";
       document.getElementById('th0').innerHTML="200 W m-2";
       document.getElementById('th1').innerHTML="400 W m-2";
       document.getElementById('th2').innerHTML="400 W m-2";
       thresholds[0]="th0200";
       thresholds[1]="th0400";
       thresholds[2]="th0400";
       kind=4;
       break;
     case 'SLOEID':
       document.getElementById('ti_memb1').innerHTML="member latent heat flux (W m-2)";
       document.getElementById('ti_memb2').innerHTML="member latent heat flux (W m-2)";
       document.getElementById('ti_mean').innerHTML="50-member ensemble mean latent heat flux (W m-2)";
       document.getElementById('ti_prob').innerHTML="SLOE probability according to threshold (1/0)";
       document.getElementById('th0').innerHTML="100 W m-2";
       document.getElementById('th1').innerHTML="200 W m-2";
       document.getElementById('th2').innerHTML="200 W m-2";
       thresholds[0]="th0100";
       thresholds[1]="th0200";
       thresholds[2]="th0200";
       kind=5;
       break;
     case 'TPID':
       document.getElementById('ti_memb1').innerHTML="member TP (mm h-1)";
       document.getElementById('ti_memb2').innerHTML="member TP (mm h-1)";
       document.getElementById('ti_mean').innerHTML="50-member ensemble mean TP (mm h-1)";
       document.getElementById('ti_prob').innerHTML="TP probability according to threshold (1/0)";
       document.getElementById('th0').innerHTML="1 mm h-1";
       document.getElementById('th1').innerHTML="5 mm h-1";
       document.getElementById('th2').innerHTML="10 mm h-1";
       thresholds[0]="th0001";
       thresholds[1]="th0005";
       thresholds[2]="th0010";
       kind=6;
       break;
     default:
       // do nothing
       break;
   }
   prepareFigure();
   return false;
}

function setSize(s)
{
  if (s==0) {
    w="300px";
  } else {
    w="500px";
  }
  document.getElementById('plot_mean').style.width=w;
  document.getElementById('plot_prob').style.width=w;
  document.getElementById('plot_mem1').style.width=w;
  document.getElementById('plot_mem2').style.width=w;
}
  

// fin
