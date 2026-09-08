# ViSuS on Streamlit Community Cloud

No ngrok, no venv, no nohup here. Community Cloud builds and runs the
whole thing for you in its own container, from a GitHub repo.

## What's in this folder

    requirements.txt   Pinned Python deps, librosa locked to >=0.10,<1.0
                        (matches the librosa.feature.tempo fix already
                        applied to app53_patched.py)
    packages.txt        System-level libs librosa/soundfile/pretty_midi
                        need underneath (ffmpeg, libsndfile1)

Add your actual app file to this folder before pushing:

    app53_patched.py

## Steps

1. Create a public GitHub repo (Community Cloud's free tier requires
   public repos).

2. Push this folder's contents plus app53_patched.py to the repo root.
   Community Cloud initializes from the repo root even if you nest
   files, so keep app53_patched.py, requirements.txt, and packages.txt
   all at the top level.

       git init
       git add .
       git commit -m "ViSuS deploy"
       git branch -M main
       git remote add origin https://github.com/rustombhesania/YOUR_REPO_NAME.git
       git push -u origin main

3. Go to https://share.streamlit.io and sign in with GitHub.

4. Click "New app", pick the repo, branch (main), and set the main
   file path to:

       app53_patched.py

5. Deploy. First build takes a few minutes while it installs
   packages.txt and requirements.txt in order.

6. Your app gets a permanent URL like:

       https://your-app-name.streamlit.app

   That's your shareable link. No separate tunnel step, no keeping a
   terminal session open on phoenix.

## If you hit the 1GB RAM ceiling

Community Cloud's free tier caps around 1GB RAM. Given the compute
load here (pyin pitch tracking, DTW across recordings), watch for the
"This app has gone over its resource limits" error. If it shows up:

  - Lean harder on st.cache_data / st.cache_resource so repeated
    Streamlit reruns don't reprocess audio from scratch
  - Downsample or shorten test audio where possible
  - If it's a structural problem (not a caching gap), Hugging Face
    Spaces is the fallback with more headroom

## Updating the app later

Push to the connected GitHub branch. Community Cloud auto-redeploys
on every push (rate limited to 5 updates/minute).
