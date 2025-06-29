; This script sets simulation parameters and calls the main procedure `synth_cme`
; to generate synthetic CME data for a specified time window.
;
; OUTPUT DIRECTORIES:
;   fitssavedir   - Directory to save synthetic FITS images
;   jpegsavedir   - Directory to save JPEG snapshots of each CME frame
;   csvsavedir    - Directory to store per-frame metadata logs
;
; SIMULATION TIME RANGE:
;   sd, ed        - Start and end times for selecting random CME timestamps
;   num_dates     - Number of synthetic CME events to generate within the range
;
; CME AND VIEWPOINT SETTINGS:
;   missions      - Missions to simulate (e.g., SOHO, STEREO)
;   spacecrafts   - Associated spacecraft (e.g., LASCO, A, B)
;   instruments   - Instruments used (e.g., C3, COR2)
;   ht0, ht1      - Height range in solar radii for CME evolution
;
; FLAGS:
;   num_iterations- Reserved for iteration control (currently unused)
;   orient_user   - Optional user-specified orientation angle flag (0b: random, 1b: use value)
;   fix_fov       - Set to 1b to fix the instrument field of view during WCS generation
;   wireframe     - Set to 1b to generate additional wireframe CME projections
;
; CALL MAIN PROCEDURE:
;   synth_cme(...) - Executes the generation pipeline with above parameters


fitssavedir = '/home/hag43/data1/fits'                                                 
jpegsavedir = '/home/hag43/data1/testim'                                                                          
csvsavedir =  '/home/hag43/data1/testcsv'                                                                          
;date = '2009/06/03 01:29:24' 
sd = '2008/01/01 00:00' 
ed = '2008/01/01 05:00'
num_dates = 1
missions = ['soho','stereo'] 
spacecrafts = ['lasco','a','b'] 
instruments = ['c3','cor2']
ht0 = 6.0 
ht1 = 16.5 
num_iterations = 1
orient_user = 1b
fix_fov=0b
wireframe = 0b

synth_cme, sd,ed, missions, spacecrafts, instruments, $
ht0, ht1, fitssavedir, csvsavedir, num_iterations, orient_user = orient_user,jpegsavedir,fix_fov=fix_fov,num_dates, wireframe = wireframe

end

