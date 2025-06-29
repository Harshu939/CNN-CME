;+
;
; NAME:
; wrapcme
;
; PURPOSE:
;
; Given a coronagraph instrument/spacecraft, choice of CME geometry (e.g. longitude, latitude relative to Earth),
; this procedure outputs a structure of density at a set of points corresponding to a CME model. The CME model is
; similar to the Croissant model, or a tube of varying radius with the two legs rooted at the Sun. The CME co-ordinates
; returned by this function are normalized to between a height of 1 and 2 solar radii. This allows other programs to 
; adapt the CME height according to a self-similar expansion.
; 
; This procedure is merely a convenient caller to make_cme function. It sets up the detailed parameter structure necessary to 
; call make_cme, allowing the user to adjust the most important parameters namely central longitude, latitude, orientation and width.
;
;
; CALLING SEQUENCE:
; out_structure=wrapcme(date,mission,sc,instr,lonrel2earth,latrel2earth,orient,width,res,twist)
;
; INPUTS:
;   Date = string giving date of required 'observation' e.g. '2011/01/01 00:00'
;
;   Mission = string: 'soho' or 'stereo'
;
;   spacecraft = string: 'a' or 'b' for STEREO, 'lasco' for SOHO
;
;   instrument = string: 'cor1' or 'cor2' for STEREO, 'c2' or 'c3' for SOHO
;
;   lonrel2earth = longitude of central position of CME relative to Earth in RADIANS
;
;   latrel2earth = latitude of central position of CME relative to Earth in RADIANS
;
;   orient = orientation of CME around its central axis in RADIANS. 90 deg means CME legs are aligned (or overlapping)
;            along the same solar latitude. 0 deg have the legs aligned on the same solar longitude
;
;   width = the maximum width of the CME tube. Arbitrary units, relative to the normalized co-ordinates of the CME height range being 1. So
;           width of 0.5 means the maximum width which is half the CME apex height measured from the photosphere.
;   
;   res = sets the 'resolution' of the CME grid, basically number of output points.
;
;   twist = main twist of the CME structure around its central (radial) axis.
;   
;   random = single number sets randomness of CME points distributed around the mean wireframe structure
;
; OUTPUTS:
;   out_structure = output IDL structure with the following fields:
;     CARRLONG        DOUBLE          -18.038668  ;Carrington longitude of central CME axis
;     CARRLAT         DOUBLE           93.006028  ;Carrington latitude of central CME axis
;     POINTS          STRUCT    -> <Anonymous> Array[1] ;see below
;     WF              STRUCT    -> <Anonymous> Array[1] ;as POINTS, but with non-randomized co-ordinates and densities. Useful for 
;                                                       ;depicting the CME structure as a wireframe model
;                                                       
;   The 'points' field in the out_structure has the following structure:
;     R               DOUBLE    Array[1200000]  ;height of each point, normalized between 1 and 2Rs
;     THE             DOUBLE    Array[1200000]  ;Carrington longitude of each point
;     PHI             DOUBLE    Array[1200000]  ;Carrington latitude of each point 
;     DENS            FLOAT     Array[1200000]  ;density of each point                                                   
;
; OPTIONAL KEYWORD
;   None
;
; OPTIONAL INPUTS
;   None

;
; OPTIONAL OUTPUTS
;   None

;
; PROCEDURE:
; As of 07/2020 the only published description of the synthetic CME codes is section 2.4 of
; https://iopscience.iop.org/article/10.1088/0004-637X/813/1/35/pdf
; We are planning a more detailed document as soon as we can.
; 
; 
;
;
; USE & PERMISSIONS
; If you use this code for synthetic CME studies, please cite https://iopscience.iop.org/article/10.1088/0004-637X/813/1/35/pdf
; Any problems/queries, or suggestions for improvements, please email Huw Morgan, hmorgan@aber.ac.uk
;
; ACKNOWLEDGMENTS:
;  This code was developed with the financial support of:
;  AberDoc scholarship Studentship to Aberystwyth University (Muro)
;  STFC Consolidated grant to Aberystwyth University (Morgan)
;
; MODIFICATION HISTORY:
; Created at Aberystwyth University 07/2019 - Huw Morgan hmorgan@aber.ac.uk
;
;-

function wrapcme,date,mission,sc,instr,lonrel2earth,latrel2earth, $
            orient,width,leg_separation,res,twist,mass,sheath_width

if strlowcase(mission) eq 'soho' then scv='Earth'
if strlowcase(mission) eq 'sdo' then scv='Earth'
if strlowcase(mission) eq 'stereo' then scv=sc
lonlatrelearth2lonlatrelobs,date,scv,lonrel2earth,latrel2earth,lonobs,latobs;all in radian!

if n_elements(res) eq 0 then res=1 

;set up main params structure. Several parameters to control the distribution of CME points, so 
;convenient to wrap up in a single structure rather than individual variables.
nmod=1.e6
params={ $
    nmod:nmod, $
    loopshapefactor:1, $
    orient:orient, $
    width:width, $
    leg_separation:leg_separation, $
    squeeze:1.0, $
    twist:twist, $ ;twist of main CME around central radial axis
    mass:mass, $
    sheath_width:sheath_width $
  }

;create the set of CME points (all centered on the solar north pole axis)
points=make_cme(params)

;then rotate CME to required central longitude and latitude
points=rotate_cme(points,lonobs,latobs)

;the following 2 lines will need attention. Arbitarily reduce density towards CME legs
htramp=[1,0.4]
htramp=mission eq 'sdo'?1:make_htramp(points,htramp[0],htramp[1], /adjust_density)

lonlatrelearth2lonlat,date,lonrel2earth,latrel2earth,lonwf,latwf

;return data structure
return,{carrlong:lonwf*!radeg,carrlat:latwf*!radeg,points:points}

end