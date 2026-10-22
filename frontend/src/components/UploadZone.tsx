import { useRef, useState } from "react";

interface UploadZoneProps {
  files: File[];
  disabled?: boolean;
  onFiles: (files: File[]) => void;
}

const ACCEPTED = [".csv", ".npy", ".npz", ".hea", ".dat"];

function validFiles(files: FileList | File[]): File[] {
  return Array.from(files).filter((file) =>
    ACCEPTED.some((extension) => file.name.toLowerCase().endsWith(extension)),
  );
}

export function UploadZone({ files, disabled, onFiles }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  return (
    <div>
      <div
        className={`upload-zone ${dragging ? "is-dragging" : ""}`}
        onClick={() => !disabled && inputRef.current?.click()}
        onDragEnter={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          if (event.currentTarget === event.target) setDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (!disabled) onFiles(validFiles(event.dataTransfer.files));
        }}
        onKeyDown={(event) => {
          if ((event.key === "Enter" || event.key === " ") && !disabled) {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Choose or drop biomedical signal files"
        aria-disabled={disabled}
      >
        <input
          ref={inputRef}
          className="visually-hidden"
          type="file"
          accept={ACCEPTED.join(",")}
          multiple
          disabled={disabled}
          onChange={(event) => {
            if (event.target.files) onFiles(validFiles(event.target.files));
            event.target.value = "";
          }}
        />
        <span className="upload-icon" aria-hidden="true">&#8593;</span>
        <strong>Drop a recording here</strong>
        <span>or choose CSV, NPY/NPZ, or a WFDB header + data pair</span>
        <small>Files stay on this computer and are never sent to a cloud service.</small>
      </div>
      {files.length > 0 && (
        <ul className="file-list" aria-label="Selected files">
          {files.map((file) => (
            <li key={`${file.name}-${file.size}`}>
              <span className="file-type">{file.name.split(".").pop()?.toUpperCase()}</span>
              <span>
                <strong>{file.name}</strong>
                <small>{file.size < 1024 ? `${file.size} B` : `${(file.size / 1024).toFixed(1)} KB`}</small>
              </span>
              <button
                type="button"
                className="icon-button"
                onClick={() => onFiles(files.filter((candidate) => candidate !== file))}
                aria-label={`Remove ${file.name}`}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
