;+
; NAME:
;   synth_cme
;
; PURPOSE:
;   Generate synthetic CME images and metadata for a specified number of events,over a given date range and for multiple spacecraft/instruments. The procedure
;   simulates flux-rope-like CMEs with randomized parameters, performs LOS integration to generate synthetic images, and saves the outputs as FITS,
;   JPEG, and CSV files. It also optionally generates wireframe projections.
;
; INPUTS:
;   sd            - Start date (string or date object; compatible with ANYTIM2TAI)
;   ed            - End date (string or date object; compatible with ANYTIM2TAI)
;   missions      - Array of missions to simulate (e.g., ['soho', 'stereo'])
;   spacecrafts   - Array of spacecraft to simulate (e.g., ['lasco', 'a', 'b'])
;   instruments   - Array of instruments (e.g., ['c3', 'cor2'])
;   ht0           - Initial CME height at the first observation time (in solar radii)
;   ht1           - Final CME height to simulate (in solar radii)
;   fitssavedir   - Directory path to save FITS images
;   csvsavedir    - Directory path to save CSV metadata files
;   jpegsavedir   - Directory path to save JPEG previews
;   num_dates     - Number of random CME events to simulate
;
; KEYWORD PARAMETERS:
;   orient_user   - Optional user-defined orientation angle for CME (degrees)
;   velkms        - Optional user-defined CME speed in km/s
;   wireframe     - If set, generate wireframe projections of the CME structure
;   num_iterations- Reserved for future use (not used in this version)
;   fix_fov       - If set, fixes the instrument field-of-view during WCS creation
;   longitude_sc  - Output longitude of spacecraft used for metadata logging
;   latitude_sc   - Output latitude of spacecraft used for metadata logging
;
; OUTPUTS:
;   - FITS files for each simulated CME snapshot saved in fitssavedir.
;   - JPEG images saved in jpegsavedir for quick visual inspection.
;   - CSV files containing simulation metadata for each snapshot.
;
; SIDE EFFECTS:
;   - Modifies environment variables for SPICE support.
;   - Generates synthetic data using random number seeds and stores image products.
;
; REQUIREMENTS:
;   - Assumes SSW (SolarSoft) environment is installed at '/home/hag43/ssw'.
;   - Requires procedures:
;     wrapcme
;     make_dummy_wcs
;     adjust_cme_height
;     make_cme_losintegration
;     writefitssynthcme
;
; EXAMPLE:
;   IDL> synth_cme '2023-01-01', '2023-01-31', ['soho','stereo'], ['lasco','a','b'], ['c3','cor2'], $
;         5.0, 15.0, fitssavedir='/data/fits', csvsavedir='/data/csv', jpegsavedir='/data/jpeg', num_dates=10, /wireframe
;
; AUTHOR:
;   Harshita Gandhi and Huw Morgan, 2025. Aberystwyth University. Adapted for synthetic CME generation with multi-view image simulation.

pro synth_cme_harshi_1, sd,ed, missions, spacecrafts, instruments, $
    ht0, ht1, orient_user=orient_user,  $
    velkms=velkms, fitssavedir, csvsavedir, wireframe = wireframe, num_iterations,jpegsavedir,fix_fov=fix_fov,$
    num_dates, longitude_sc=longitude_sc,latitude_sc=latitude_sc

; Define the range for random parameter generation
sinlat_range = [-1,1] ; sine of latitude range for bias removal
lon_range = [0, 360] ; Longitude range in degrees
velocity_range = [200, 3000] ; Velocity range in km/s
orient_range = [-90,90]
stai = anytim2tai(sd)
etai = anytim2tai(ed)


homessw='/home/hag43/ssw'        ;change this directory where your ssw is installed
setenv, 'SPICE_ICY_DLM='+homessw+'/icy/lib/icy.dlm'
spice_soho=homessw+'/soho/gen/data/spice'
setenv,'SOHO_SPICE='+spice_soho
setenv,'SOHO_SPICE_GEN='+spice_soho

seed = -1

; Define cadences
lasco_c3_cadence= 12*60.
cor2_cadence = 30*60.     


; Main loop to iterate over random dates
for date_index = 0, num_dates-1 do begin
  rtai = stai+randomu(seed,1)*(etai-stai)
  date = anytim2cal(rtai, form  = 11)
  
  date_c3 = date
  date_cor2a = anytim2cal(anytim2tai(date) - 10, form=11) ; COR2A starts 6 minutes earlier
  date_cor2b = anytim2cal(anytim2tai(date_cor2a) + 21, form=11) ; COR2B also starts 6 minutes earlier
    lonrel2earth = randomu(seed) * (lon_range[1] - lon_range[0]) + lon_range[0]
    sinlatrel2earth = randomu(seed) * (sinlat_range[1] - sinlat_range[0]) + sinlat_range[0]
    latrel2earth = asin(sinlatrel2earth) * !radeg
    velkms = randomu(seed) * (velocity_range[1] - velocity_range[0]) + velocity_range[0]
    orient_user = randomu(seed) * (orient_range[1] - orient_range[0]) + orient_range[0]
    
    lonrel2earth *= !dtor
    latrel2earth *= !dtor
    orient=keyword_set(orient_user)?orient_user*!dtor:0*!dtor
    res = 1 ; resolution
    width = 0.1 ; cylinder width
    twist = 0.
    mass = 1.d15 ; grams total CME mass
    sheath_width = 0.0 ; width of sheath
    leg_separation = 0.6


    ; Define the CME index
    cme_index = 'cme' 

    
      ; Initialize array to store log entries
      log_entries = []
      lon_soho = ''
      lat_soho = ''
      lon_sta = ''
      lat_sta = ''
      lon_stb = ''
      lat_stb = ''
      pixperrs_soho = ''
      pixperrs_sta = ''
      pixperrs_stb = ''
    
    ; Loop over each combination of mission, spacecraft, and instrument
      foreach mission, missions do begin
          if mission eq 'soho' then begin
              instrument = 'c3'
              spacecraft_list = ['lasco']           
              cadence = lasco_c3_cadence
              date_initial = date_c3
              ht0_initial = ht0 ; Initial height for LASCO C3
              print, 'LASCO/C3 Initial Height: ', ht0_initial
          endif else if mission eq 'stereo' then begin
              instrument = 'cor2'
              spacecraft_list = ['a', 'b']
              cadence = cor2_cadence
          endif

          foreach spacecraft, spacecraft_list do begin
            ; Assign initial observation time and initial height based on spacecraft
              if spacecraft eq 'a' then begin
                date_initial = date_cor2a
                time_diff = anytim2tai(date_cor2a) - anytim2tai(date_c3) ; Time difference in seconds
                print, 'COR2A Time Diff (seconds): ', time_diff
                ht0_initial = ht0 + (velkms * time_diff / rsunkm)
                print, 'COR2A Initial Height: ', ht0_initial, ' velkms: ', velkms, ' rsunkm: ', rsunkm
;                ht0_initial = 5.0; Initial height for COR2A
              endif else if spacecraft eq 'b' then begin
                date_initial = date_cor2b
                time_diff = anytim2tai(date_cor2b) - anytim2tai(date_c3) ; Time difference in seconds
                ht0_initial = ht0 + (velkms * time_diff / rsunkm) ; Adjust height for COR2B
              endif
             print, 'Mission: ', mission, ' Spacecraft: ', spacecraft, $
          ' Initial Time: ', date_initial, $
          ' Initial Height: ', ht0_initial
            ; Call the CME synthesis code
            
            points = wrapcme(date_initial, mission, spacecraft, instrument, lonrel2earth, latrel2earth, orient, $
                            width, leg_separation, res, twist, mass, sheath_width)
                     
            wcs = make_dummy_wcs(date, mission, spacecraft, instrument, mask = mask, htra = htra, $
                            rsun_arcsec = rsun_arcsec,fix_fov=fix_fov,longitude_sc=longitude_sc,latitude_sc=latitude_sc)
                         
            ; Store the longitude and latitude values for the current spacecraft
            if spacecraft eq 'lasco' then begin
               lon_soho = strtrim(longitude_sc, 2)
               lat_soho = strtrim(latitude_sc, 2)
            endif else if spacecraft eq 'a' then begin
               lon_sta = strtrim(longitude_sc, 2)
               lat_sta = strtrim(latitude_sc, 2)
            endif else if spacecraft eq 'b' then begin
               lon_stb = strtrim(longitude_sc, 2)
               lat_stb = strtrim(latitude_sc, 2)
            endif      
        
            ; Calculate pixel size for the current spacecraft
              pixperrs = rsun_arcsec / wcs.cdelt[0]       
                
            
              indnan = where(~mask)
              rsunkm = wcs_rsun(units = 'km')
              nobs = ceil((ht1 - ht0_initial) * rsunkm / (velkms * cadence))
              ht = make_coordinates(nobs, [ht0_initial, ht1])
            
              nxjpg = 256
              nyjpg = 256
              set_plot, 'Z'
              device, set_resolution = [nxjpg, nyjpg], Set_Pixel_Depth = 24, Decomposed = 0, z_buffer = 0
              ctload, 0
              device, decomposed = 0
              !p.background = 255 & !p.color = 0
              noise = 1.e-4
          
              for iht = 0, n_elements(ht) - 1 do begin
                
                  datenow = anytim2cal(anytim2tai(date_initial) + cadence * iht, form = 11)                
                  adjust_cme_height, points.points, ht[iht], xn, yn, zn, xnwf, ynwf, znwf
                  im = make_cme_losintegration(wcs, xn, yn, zn, points.points.dens)
                  ixy = keyword_set(wireframe) ? make_cme_losintegration(wcs, xnwf, ynwf, znwf, /returnxy) : !null
                
                  noise = randomn(seed, wcs.naxis[0], wcs.naxis[1]) * noise
                  im[indnan] = !values.f_nan
                  fitsfilename = fitssavedir + '/' + cme_index + '_' + strlowcase(instrument) + '_' + strlowcase(spacecraft) + '_' + $
                              strlowcase(mission) + '_dynamics_' + strmid(anytim2cal(datenow, form = 8), 0, 8) + '_' + $
                              strmid(anytim2cal(datenow, form = 8), 8, 6) + '.fits'
                
                  writefitssynthcme, '', datenow, mission, spacecraft, instrument, im, wcs.cdelt[0], wcs.crpix[0], wcs.crpix[1], htra, fitsfilename, rsun_arcsec, /no_convert_pixsize
                  jpegfilename = jpegsavedir + '/' + cme_index + '_' + strlowcase(instrument) + '_' + strlowcase(spacecraft) + '_' + $
                               strlowcase(mission) + '_dynamics_' + strmid(anytim2cal(datenow, form = 8), 0, 8) + '_' + $
                               strmid(anytim2cal(datenow, form = 8), 8, 6) + '.jpeg'              
               
                  csv = csvsavedir + '/' + cme_index + '_' + $
                               strmid(anytim2cal(datenow, form = 8), 0, 8) + '_' + $
                               strmid(anytim2cal(datenow, form = 8), 8, 6) + '.csv'
                  filename =   cme_index + '_'  + $
                               strmid(anytim2cal(datenow, form = 8), 0, 8) + '_' + $
                               strmid(anytim2cal(datenow, form = 8), 8, 6)
                  openw, lun_csv, csv, /get_lun
                  printf, lun_csv, 'filename,lon_soho,lat_soho,pixperrs_soho,lon_sta,lat_sta,pixperrs_sta,lon_stb,lat_stb,pixperrs_stb,LonRel2Earth,LatRel2Earth,Orient,Height,Speed'

                mnmx = minmax(im, /nan)
                ind = where(~finite(im), cntnan, comp = nind, ncomp = ncntnan)
                ctload, 0
                
                ; Check min and max values before hist_equal
                minv = min(im, /nan)
                maxv = max(im, /nan)
                print, 'Min value: ', minv
                print, 'Max value: ', maxv

                ; Ensure minv < maxv
                if minv lt maxv then begin
                  imjpeg = hist_equal(im, per = 0.1)
                endif else begin
                  print, 'Skipping hist_equal due to invalid min/max values'
                  imjpeg = im  ; Or handle this case as appropriate
                end
                               

                if cntnan gt 0 then imjpeg[ind] = 0
                imjpeg = congrid(imjpeg, nxjpg, nyjpg, /interp)
                sizearr, im, nx, ny
                
                if iht eq 0 then begin
                  pixfactor=nxjpg/float(nx)
                  pixperrs=pixperrs*pixfactor

                  ; Store the longitude and latitude values for the current spacecraft
                  case spacecraft of
                    'lasco':begin
                      lon_soho = strtrim(longitude_sc, 2)
                      lat_soho = strtrim(latitude_sc, 2)
                      pixperrs_soho = strtrim(pixperrs, 2)
                    end
                    'a':begin
                      lon_sta = strtrim(longitude_sc, 2)
                      lat_sta = strtrim(latitude_sc, 2)
                      pixperrs_sta = strtrim(pixperrs, 2)
                    end
                    'b':begin
                      lon_stb = strtrim(longitude_sc, 2)
                      lat_stb = strtrim(latitude_sc, 2)
                      pixperrs_stb = strtrim(pixperrs, 2)
                    end
                  endcase
                endif


                snapshot = tvrd(/true)
                write_jpeg, jpegfilename, snapshot, true = 1, quality = 100
           

                ; Calculate height at current time
                current_height = ht0_initial + (velkms * (cadence * iht) / rsunkm)
                  
              log_entry_i = filename + ',' + lon_soho + ',' + lat_soho + ',' + pixperrs_soho + ',' + $
              lon_sta + ',' + lat_sta + ',' + pixperrs_sta + ',' + $
              lon_stb + ',' + lat_stb + ',' + pixperrs_stb + ',' + $
              strjoin([strtrim(lonrel2earth * !radeg, 2), strtrim(latrel2earth * !radeg, 2), $
              strtrim(orient * !radeg, 2), strtrim(current_height, 2), strtrim(velkms, 2)], ', ')

                printf, lun_csv, log_entry_i                 
                
                free_lun, lun_csv

                
            endfor;height
        endforeach;spacecraft


     endforeach

  endfor


print,date
setx_corimp, /cle


end
