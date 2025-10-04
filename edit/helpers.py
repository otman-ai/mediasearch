import os
import ffmpeg
import cv2

def compression(input_file, output_file):
    (
        ffmpeg
        .input(input_file)
        .output(output_file, vcodec="libx264", crf=28)
        .overwrite_output()
        .run()
    )

def add_audio(video_file, audio_file, output_file):
    input_video = ffmpeg.input(video_file)
    input_audio = ffmpeg.input(audio_file)
    (
        ffmpeg
        .output(input_video.video, 
                input_audio.audio, 
                output_file, 
                vcodec='copy', 
                acodec='aac')
        .overwrite_output()
        .run()
    )

def blur_region(image, x, y, w, h):
    roi = image[y:h, x:w]
    roi = cv2.GaussianBlur(roi, (51, 51), 30)  # strong blur
    image[y:h, x:w] = roi
    return image

def get_video_duration(path):
    probe = ffmpeg.probe(path)
    duration = float(probe['format']['duration'])
    return duration

def extract_audio(input_video, output_audio):
    (
        ffmpeg
        .input(input_video)
        .output(output_audio, **{'q:a': 0}, map='a')  # high quality audio, map only audio
        .overwrite_output()

        .run()
    )

def cut_video(input_file, output_file, start_time, duration, reencode=True):
    if reencode:
        (
            ffmpeg
            .input(input_file, ss=start_time, t=duration)
            .output(output_file, vcodec="libx264", acodec="aac")
            .overwrite_output()
            .run()
        )
    else:
        (
            ffmpeg
            .input(input_file, ss=start_time, t=duration)
            .output(output_file, c="copy")
            .overwrite_output()
            .run()
        )

def remove_audio(input_file, output_file):
    (
        ffmpeg
        .input(input_file)
        .output(output_file, c="copy", an=None)  # an = no audio
        .overwrite_output()
        .run()
    )

def remove_intervals(video_path, remove_intervals, duration, output_path, reencode=True):
    remove_intervals = sorted(remove_intervals)
    
    # Compute keep intervals (complement)
    keep_intervals = []
    last = 0
    for start, end in remove_intervals:
        if start > last:
            keep_intervals.append((last, start))
        last = end
    if last < duration:
        keep_intervals.append((last, duration))

    print("Keeping intervals:", keep_intervals)

    # Use the cut_intervals function logic
    parts = []
    temp_dir = os.path.join(os.path.expanduser("~"), ".cache")
    for i, (start, end) in enumerate(keep_intervals):
        part_file = os.path.join(temp_dir, f"keep_{i}.mp4")
        d = end - start
        
        cut_video(video_path, part_file, start, d, reencode)

        parts.append(part_file)

    # Make concat list
    files_dir = os.path.join(temp_dir, 'inputs.txt')
    with open(files_dir, "w") as f:
        for p in parts:
            f.write(f"file '{os.path.abspath(p)}'\n")

    # Concatenate
    if not reencode:
        (
            ffmpeg
            .input(files_dir, format="concat", safe=0)
            .output(output_path, c="copy")
            .overwrite_output()
            .run()
        )
    else:
        (
            ffmpeg
            .input(files_dir, format="concat", safe=0)
            .output(output_path)
            .overwrite_output()
            .run()
        )
    for p in parts:
        os.remove(p)
    os.remove(files_dir)
