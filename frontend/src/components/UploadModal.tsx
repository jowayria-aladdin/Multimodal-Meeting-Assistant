"use client";

import { useState, useRef } from "react";
import { X, Mic, Video, FileVideo, Loader2 } from "lucide-react";
import { apiFetch } from "@/lib/api";

interface Props {
  onClose: () => void;
  onSuccess: () => void;
}

export default function UploadModal({ onClose, onSuccess }: Props) {
  const [uploadData, setUploadData] = useState({ title: "", language: "Arabic" });
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [signFile, setSignFile] = useState<File | null>(null);
  
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgressText, setUploadProgressText] = useState("");
  const [uploadProgressPercent, setUploadProgressPercent] = useState(0);

  // The kill switch reference
  const abortControllerRef = useRef<AbortController | null>(null);

  // Helper function to upload large files in chunks directly to Cloudinary
  const uploadToCloudinaryChunked = async (file: File, cloudName: string, uploadPreset: string, signal: AbortSignal) => {
    const CHUNK_SIZE = 5 * 1024 * 1024; 
    const url = `https://api.cloudinary.com/v1_1/${cloudName}/auto/upload`;
    
    const uniqueUploadId = `lughacap_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
    let currentChunkStart = 0;
    let finalResponse = null;

    while (currentChunkStart < file.size) {
      const currentChunkEnd = Math.min(currentChunkStart + CHUNK_SIZE, file.size);
      const chunk = file.slice(currentChunkStart, currentChunkEnd);

      const formData = new FormData();
      formData.append("file", chunk);
      formData.append("upload_preset", uploadPreset);

      const contentRange = `bytes ${currentChunkStart}-${currentChunkEnd - 1}/${file.size}`;
      
      // Attach the abort signal to the fetch request
      const res = await fetch(url, {
        method: "POST",
        headers: {
          "X-Unique-Upload-Id": uniqueUploadId,
          "Content-Range": contentRange,
        },
        body: formData,
        signal: signal,
      });

      if (!res.ok) {
        throw new Error(`Cloudinary upload failed during chunk ${currentChunkStart}-${currentChunkEnd}`);
      }

      finalResponse = await res.json();
      currentChunkStart = currentChunkEnd;
      
      const percent = Math.floor((currentChunkEnd / file.size) * 75); 
      setUploadProgressPercent(percent);
    }

    return finalResponse; 
  };

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!videoFile || !audioFile || !signFile) {
      alert("Please ensure all three media files (Audio, Main Video, Sign Video) are selected.");
      return;
    }

    setIsUploading(true);
    setUploadProgressPercent(0);

    // Initialize the kill switch for this upload
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const mainMB = (videoFile.size / (1024 * 1024)).toFixed(2);
    
    try {
      setUploadProgressText(`Uploading main video (${mainMB}MB)...`);
      const cloudName = process.env.NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME;
      const uploadPreset = process.env.NEXT_PUBLIC_CLOUDINARY_PRESET;

      if (!cloudName || !uploadPreset) {
        throw new Error("Cloudinary credentials are missing from environment variables.");
      }

      //Upload to Cloudinary using Chunks (passing the kill switch signal)
      const cloudinaryJson = await uploadToCloudinaryChunked(videoFile, cloudName, uploadPreset, controller.signal);
      
      if (!cloudinaryJson || !cloudinaryJson.secure_url) {
          throw new Error("Failed to retrieve secure URL from Cloudinary.");
      }

      const mainVideoUrl = cloudinaryJson.secure_url;
      const mainVideoPublicId = cloudinaryJson.public_id;

      setUploadProgressText("Sending data to the backend server...");
      setUploadProgressPercent(85); 
      
      let langCode = "ar"; 
      if (uploadData.language === "English") langCode = "en";
      if (uploadData.language === "Code-switch") langCode = "cs";

      // 2. Prepare Backend Payload
      const backendFormData = new FormData();
      backendFormData.append("title", uploadData.title);
      backendFormData.append("lang", langCode);
      backendFormData.append("mainVideoUrl", mainVideoUrl);
      backendFormData.append("mainVideoPublicId", mainVideoPublicId);
      backendFormData.append("signVideo", signFile); 
      backendFormData.append("wavFile", audioFile);  

      // Send to Backend using apiFetch (passing the kill switch signal)
      await apiFetch("/meetings/upload", {
        method: "POST",
        data: backendFormData,
        signal: controller.signal
      });

      setUploadProgressPercent(100);
      setUploadData({ title: "", language: "Arabic" });
      setAudioFile(null); setVideoFile(null); setSignFile(null);
      
      onSuccess(); 

    } catch (err: unknown) {
      if (err instanceof Error && (err.name === "AbortError" || err.message.includes("aborted"))) {
        console.log("Upload cancelled by user.");
        return;
      }
      console.error(err);
      alert(`Upload Failed: ${err instanceof Error ? err.message : "Unknown Error"}`);
    } finally {
      setIsUploading(false);
      setUploadProgressText("");
      setUploadProgressPercent(0);
      abortControllerRef.current = null;
    }
  };

  //to trigger the abort before closing the modal
  const handleCancelAndClose = () => {
    if (isUploading && abortControllerRef.current) {
      abortControllerRef.current.abort(); // FIRE THE KILL SWITCH
    }
    onClose(); // Hide the UI
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm">
      <div className="bg-white w-full max-w-2xl rounded-3xl shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200 flex flex-col max-h-[90vh]">
        
        <div className="px-8 py-5 border-b border-slate-100 flex items-center justify-between shrink-0">
          <div>
            <h3 className="font-serif text-xl font-bold text-slate-900">Upload Meeting Resources</h3>
            <p className="text-sm text-slate-500 mt-1">Provide the media files for AI processing and translation.</p>
          </div>
          {/*Call handleCancelAndClose */}
          <button 
            type="button"
            onClick={handleCancelAndClose} 
            className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-full transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        <div className="p-8 overflow-y-auto relative">
            {isUploading && (
              <div className="absolute inset-0 z-10 bg-white/80 backdrop-blur-sm flex flex-col items-center justify-center p-8 text-center rounded-b-3xl">
                <Loader2 className="w-12 h-12 text-brand-maroon animate-spin mb-4" />
                <h4 className="text-lg font-bold text-slate-900 mb-4">{uploadProgressText}</h4>
                
                <div className="w-full max-w-xs flex items-center justify-between text-sm font-bold text-slate-700 mb-1.5">
                  <span>Uploading...</span>
                  <span className="text-brand-maroon">{uploadProgressPercent}%</span>
                </div>
                
                <div className="w-full max-w-xs bg-slate-200 rounded-full h-3 mb-2 overflow-hidden shadow-inner">
                  <div className="bg-brand-maroon h-3 rounded-full transition-all duration-300" style={{ width: `${uploadProgressPercent}%` }}></div>
                </div>
                <p className="text-sm font-medium text-slate-500 mt-2">Please do not close this window or tab.</p>
              </div>
            )}

          <form id="uploadForm" onSubmit={handleUploadSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Meeting Title *</label>
                <input 
                  type="text" 
                  required 
                  value={uploadData.title} 
                  onChange={(e) => setUploadData({...uploadData, title: e.target.value})} 
                  placeholder="e.g., Q1 Planning Sync" 
                  disabled={isUploading}
                  className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:border-brand-gold focus:ring-1 focus:ring-brand-gold transition-colors disabled:opacity-50"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Source Language</label>
                <select 
                  value={uploadData.language} 
                  onChange={(e) => setUploadData({...uploadData, language: e.target.value})} 
                  disabled={isUploading}
                  className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:border-brand-gold focus:ring-1 focus:ring-brand-gold transition-colors text-slate-700 bg-white disabled:opacity-50"
                >
                  <option value="Arabic">Arabic</option>
                  <option value="English">English</option>
                  <option value="Code-switch">Code-switch</option>
                </select>
              </div>
            </div>

            <hr className="border-slate-100" />

            <div className="space-y-4">
              <h4 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-2">Required Media</h4>
              
              <div className="flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-xl border border-slate-200 bg-slate-50 hover:border-brand-gold/50 transition-colors gap-4">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center shrink-0">
                    <Mic size={20} />
                  </div>
                  <div>
                    <p className="font-medium text-slate-900 text-sm">Raw Audio (ASR)</p>
                    <p className="text-xs text-slate-500">.wav format required</p>
                  </div>
                </div>
                <input 
                  type="file" 
                  required 
                  accept=".wav" 
                  onChange={(e) => setAudioFile(e.target.files?.[0] || null)} 
                  disabled={isUploading}
                  className="text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer w-full sm:w-48 disabled:opacity-50"
                />
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-xl border border-slate-200 bg-slate-50 hover:border-brand-gold/50 transition-colors gap-4">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-full bg-brand-maroon/10 text-brand-maroon flex items-center justify-center shrink-0">
                    <Video size={20} />
                  </div>
                  <div>
                    <p className="font-medium text-slate-900 text-sm">Main Meeting Video</p>
                    <p className="text-xs text-slate-500">.webm or .mp4 format</p>
                  </div>
                </div>
                <input 
                  type="file" 
                  required 
                  accept=".webm,.mp4" 
                  onChange={(e) => setVideoFile(e.target.files?.[0] || null)} 
                  disabled={isUploading}
                  className="text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-brand-maroon/10 file:text-brand-maroon hover:file:bg-brand-maroon/20 cursor-pointer w-full sm:w-48 disabled:opacity-50"
                />
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-xl border border-slate-200 bg-slate-50 hover:border-brand-gold/50 transition-colors gap-4">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-full bg-brand-gold/20 text-yellow-700 flex items-center justify-center shrink-0">
                    <FileVideo size={20} />
                  </div>
                  <div>
                    <p className="font-medium text-slate-900 text-sm">Sign Language Video</p>
                    <p className="text-xs text-slate-500">.webm format required</p>
                  </div>
                </div>
                <input 
                  type="file" 
                  required 
                  accept=".webm" 
                  onChange={(e) => setSignFile(e.target.files?.[0] || null)} 
                  disabled={isUploading}
                  className="text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-brand-gold/20 file:text-yellow-800 hover:file:bg-brand-gold/30 cursor-pointer w-full sm:w-48 disabled:opacity-50"
                />
              </div>
            </div>
          </form>
        </div>

        <div className="px-8 py-4 border-t border-slate-100 bg-slate-50 flex items-center justify-end gap-3 shrink-0">
          {/*  Call handleCancelAndClose */}
          <button 
            type="button" 
            onClick={handleCancelAndClose} 
            className="px-5 py-2.5 text-sm font-medium text-slate-600 hover:text-slate-900 transition-colors"
          >
            Cancel
          </button>              
          <button 
            form="uploadForm" 
            type="submit" 
            disabled={isUploading || !uploadData.title.trim()} 
            className="px-6 py-2.5 bg-brand-maroon text-white text-sm font-medium rounded-lg hover:bg-brand-gold transition-colors flex items-center gap-2 disabled:opacity-70"
          >
            {isUploading ? <><Loader2 size={16} className="animate-spin" /> Processing...</> : "Process Recording"}
          </button>
        </div>
        
      </div>
    </div>
  );
}