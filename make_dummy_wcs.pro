function make_dummy_wcs,date,mission,spacecraft,instr, $
                    mask=mask,htra=htra,rsun_arcsec=rsun_arcsec,fix_fov=fix_fov, $
                    longitude_sc=longitude_sc,latitude_sc=latitude_sc

if strlowcase(mission) eq 'soho' then scv='soho'
if strlowcase(mission) eq 'sdo' then scv='sdo'
if strlowcase(mission) eq 'stereo' then scv=spacecraft

case strupcase(instr+scv) of
  'AIASDO':begin
    cdelt=2.4;arcsec
    nxy=1024
    obs='earth'
    minht=0.001
    maxht=1.4
   end
  'COR2A':begin
    cdelt=117.6
    nxy=256;beacon data
    obs='sta'
    minht=3.5
    maxht=15.
   end
  'COR2B':begin
    cdelt=117.6
    nxy=256;beacon data
    obs='stb'
    minht=3.5
    maxht=15.
  end
;  'C3SOHO':begin
;    cdelt=56.
;    nxy=1024
;    obs='soho'
;    minht=3.5
;    maxht=15.
   'C3SOHO':begin
     cdelt=224.
     nxy=256
     obs='soho'
     minht=3.5
     maxht=15.
  end
  'C2SOHO':begin
    cdelt=11.9
    nxy=1024
    obs='soho'
    minht=2.2
    maxht=6.0
  end
  else:stop
endcase

if keyword_set(fix_fov) then begin
  cdelt=100
  nxy=512
  minht=4.
  maxht=20.
endif

obspos=get_sunspice_lonlat(date,obs,system='Carrington',/deg,/meter)

longitude_sc=obspos[1]
latitude_sc=obspos[2]
wcs=wcs_2d_simulate(nxy,cdelt=cdelt,date_obs=date, dsun_obs=obspos[0],crln_obs=obspos[1],crlt_obs=obspos[2])
c = wcs_get_coord(wcs)

rsun_arcsec=asin(wcs_rsun()/wcs.position.dsun_obs)*!radeg*3600
c=c/rsun_arcsec
c=sqrt(total(c^2,1))
mask=c gt minht and c lt maxht
htra=[minht,maxht]

return,wcs

end