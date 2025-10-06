import os
import ffmpeg
import cv2

def compression(input_file, output_file):
    """Compress video file.
        Args:
            input_file: the video you want to compress.
            output_file: the path where you want to be the compressed file.
        """
    (
        ffmpeg
        .input(input_file)
        .output(output_file, vcodec="libx264", crf=28)
        .overwrite_output()
        .run()
    )
    return True

def add_audio(video_file, audio_file, output_file):
    """Add audio to video file.
        Args:
            video_file: the video file you want to add the audio to.
            audio_file: the path of the audio you want to add to the video.
            output_file: the path where you want to export the final video to .
        """
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
    return True

def blur_region(image, x, y, w, h):
    """Blur region in the image.
        Args:
            image: nd array image type
            x: the coordintate x 
            y: the coordinatate y.
            w: the width of the blure.
            h: the height of the blure
        Return:
            image: image blurred in nd array."""
    roi = image[y:h, x:w]
    roi = cv2.GaussianBlur(roi, (51, 51), 30)  # strong blur
    image[y:h, x:w] = roi
    return image

def get_video_duration(path):
    """Get the video duration in seconds:
        Args:
            path: the video path
        Returns:
            duration: the duration of the video in seconds.
    """
    probe = ffmpeg.probe(path)
    duration = float(probe['format']['duration'])
    return duration

def extract_audio(input_video, output_audio):
    """Extract the audio from the video.
        Args: 
            input_video: the path to the video you want to extract the audio from.
            output_audio: the patht to export the audio to.
        """
    (
        ffmpeg
        .input(input_video)
        .output(output_audio, **{'q:a': 0}, map='a')  # high quality audio, map only audio
        .overwrite_output()

        .run()
    )
    return True

def cut_video(input_file, output_file, start_time, duration, reencode=True):
    """Cut the video based on the start time and the duration.
        Args:
            input_file: the path to the video you want to cut.
            output_file: the path where you want to export the video to .
            start_time: where you want your new video to start in seconds.
            duration: the duration in seconds of the new video.
        """
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
    return True

def remove_audio(input_file, output_file):
    """Remove the audio from the video.
        Args:
            input_file: the path of the video you want to remove the audio from.
            output_file: where you want to export the video.
        """
    (
        ffmpeg
        .input(input_file)
        .output(output_file, c="copy", an=None)  # an = no audio
        .overwrite_output()
        .run()
    )
    return True

def remove_intervals(video_path, remove_intervals, duration, output_path, reencode=True):
    """Remove range of clips from the video.
        Args:
            video_path: the path of the video you want to remove the clips from.
            remove_intervals: [(start, end), (start2, end2)] the intervals of the clips.
            duration: the duration of the original video.
            output_path: where you want to export the video to.
        """
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

    return True
