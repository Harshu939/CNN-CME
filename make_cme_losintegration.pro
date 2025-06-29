function make_cme_losintegration,wcs,xn0,yn0,zn0,dens0,euv=euv,ninimage=ninimage,returnxy=returnxy
      
minrs=phys_constants('si',/rsun)

ht=sqrt(xn0^2+yn0^2+zn0^2)
xn = xn0*minrs
yn = yn0*minrs
zn = zn0*minrs

im=fltarr(wcs.naxis[0],wcs.naxis[1])

; Conversion factor from radian to degree and relavent unit arcsec etc
case wcs.cunit[0] of
  'arcmin': conv = !radeg * 60.d0
  'arcsec': conv = !radeg * 3600.d0
  'mas':    conv = !radeg * 3600.d3
  'rad':    conv = 1.d0
  else:   
endcase
; Convert to HPC
d = sqrt(yn^2 + zn^2 + (wcs.position.dsun_obs - xn)^2)
thetax = atan(yn, (wcs.position.dsun_obs - xn) )*conv
thetay = asin(zn/d)*conv

ixiy = wcs_get_pixel(wcs, transpose([[thetax],[thetay]]))
if keyword_set(returnxy) then return,ixiy

hist=hist_nd(ixiy,[1,1],min=[0,0],max=[wcs.naxis[0]-1,wcs.naxis[1]-1],rev=ri)
ind=where(hist ne 0,cnt)
ninimage=0

if ~keyword_set(euv) then begin
  pos=sqrt(yn0^2+zn0^2)
  dens=dens0*makeg(pos,ht,/bk)
endif else begin
  dens=dens0^2
endelse

for i=0,cnt-1 do begin
  indnow=get_rev_ind(ri,ind[i],nnow)
  im[ixiy[0,indnow],ixiy[1,indnow]]=total(dens[indnow],/nan)
  ninimage=ninimage+nnow
endfor

return,im


end
