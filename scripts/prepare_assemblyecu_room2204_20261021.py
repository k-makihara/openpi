"""Merge six already-cleaned room2204 sources with the curated five-source dataset.

Preserves numeric state/action values; relabels room2204 with the approved ECU
instruction. Uses the established OpenPI PIL resize, then H264 CRF18 fixed GOP12.
Publishes info.json and pipeline_complete.json only after all videos validate.
"""
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import shutil

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from resize_lerobot_videos import signature, write_json
from openpi_client import image_tools

ROOT = Path('/fsx/koshi.makihara')
REFERENCE = ROOT / 'datasets_tasked/lerobot_v21_assemblyecu_5sources_yubi_20261017_224_gop12'
OUTPUT = ROOT / 'datasets_tasked/lerobot_v21_assemblyecu_room2204_6days_plus_5sources_20261021_224_gop12'
DATES = ['20261006', '20261007', '20261008', '20261009', '20261013', '20261014']
SOURCES = [ROOT / 'datasets' / f'lerobot_v21_ph2_success_{date}_AssemblyECU_room2204_clean' for date in DATES] + [REFERENCE]
KEYS = ['observation.image.left', 'observation.image.right']


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def jsonl(path, data):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in data))


def relative(info, kind, index, key=None):
    return Path(info[kind].format(episode_index=index, episode_chunk=index // info['chunks_size'], video_key=key))


def video(job):
    source, target, expected = job
    receipt = target.with_suffix('.merge-video.json')
    source_sig = signature(source)
    if target.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['source'] == source_sig and saved['output'] == signature(target):
            return saved
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.partial.mp4')
    # Videos from the reference are already encoded as requested. Copy them
    # without adding another lossy generation, but validate every decoded frame.
    if source.is_relative_to(REFERENCE):
        shutil.copyfile(source, temporary)
    else:
        with av.open(str(source)) as reader, av.open(str(temporary), 'w', options={'movflags': '+faststart'}) as writer:
            stream = reader.streams.video[0]
            stream.codec_context.thread_count = 1
            assert stream.average_rate == 60
            encoded = writer.add_stream('libx264', rate=stream.average_rate)
            encoded.width = encoded.height = 224
            encoded.pix_fmt = 'yuv420p'
            encoded.time_base = encoded.codec_context.time_base = stream.time_base
            encoded.codec_context.thread_count = 1
            encoded.codec_context.gop_size = 12
            encoded.codec_context.max_b_frames = 0
            encoded.options = {'crf': '18', 'preset': 'fast', 'x264-params': 'keyint=12:min-keyint=12:scenecut=0:bframes=0'}
            count = 0
            for frame in reader.decode(video=0):
                assert frame.pts is not None
                assert abs(float(frame.pts * frame.time_base) - count / 60) < 1e-5
                rgb = image_tools.resize_with_pad(frame.to_ndarray(format='rgb24'), 224, 224)
                out = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                out.pts, out.time_base = frame.pts, frame.time_base
                for packet in encoded.encode(out):
                    writer.mux(packet)
                count += 1
            for packet in encoded.encode():
                writer.mux(packet)
            assert count == expected, (source, count, expected)
    with av.open(str(temporary)) as reader:
        stream = reader.streams.video[0]
        stream.codec_context.thread_count = 1
        assert (stream.width, stream.height, stream.average_rate, stream.codec_context.format.name) == (224, 224, 60, 'yuv420p')
        count = 0
        for frame in reader.decode(video=0):
            assert abs(float(frame.pts * frame.time_base) - count / 60) < 1e-5
            assert bool(frame.key_frame) == (count % 12 == 0), (source, count)
            count += 1
        assert count == expected, (source, count, expected)
    temporary.replace(target)
    saved = {'source_path': str(source), 'source': source_sig, 'output': signature(target), 'frames': count, 'gop': 12, 'timestamps_verified': True}
    write_json(receipt, saved)
    return saved


def main():
    meta = OUTPUT / 'meta'
    meta.mkdir(parents=True, exist_ok=True)
    if (meta / 'pipeline_complete.json').exists():
        print('Already complete:', OUTPUT, flush=True)
        return
    info = json.loads((REFERENCE / 'meta/info.json').read_text())
    tasks = rows(REFERENCE / 'meta/tasks.jsonl')
    assembly = next(row for row in tasks if row['task'] != 'Stay')
    task_id = assembly['task_index']
    episodes, statistics, mapping, videos, source_summary = [], [], [], [], []
    offset = 0
    for source in SOURCES:
        src_info = json.loads((source / 'meta/info.json').read_text())
        assert src_info['fps'] == 60
        for key, feature in info['features'].items():
            if 'image' not in key:
                assert src_info['features'][key] == feature, (source, key)
        build = json.loads((source / '.training-build.json').read_text())
        assert build['settings']['position_unit'] == 'mm'
        assert build['gripper_transform']['status'] == 'applied'
        assert build['gripper_transform']['implementation_sha256'] == json.loads((REFERENCE / '.training-build.json').read_text())['gripper_transform']['implementation_sha256']
        source_rows = rows(source / 'meta/episodes.jsonl')
        assert len(source_rows) == src_info['total_episodes']
        stats_by_id = {row['episode_index']: row for row in rows(source / 'meta/episodes_stats.jsonl')}
        source_frames = 0
        for row in source_rows:
            old, new = row['episode_index'], len(episodes)
            table = pq.read_table(source / relative(src_info, 'data_path', old))
            length = table.num_rows
            assert length == row['length']
            n_tasks = np.full(length, task_id) if source != REFERENCE else np.asarray(table['task_index'].to_pylist())
            assert set(n_tasks.tolist()) <= {r['task_index'] for r in tasks}
            updates = {'episode_index': np.full(length, new), 'index': np.arange(offset, offset + length), 'task_index': n_tasks, 'short_horizon_task_index': n_tasks}
            for name, values in updates.items():
                if name in table.column_names:
                    table = table.set_column(table.schema.get_field_index(name), name, pa.array(values, type=table.schema.field(name).type))
            assert np.allclose(np.asarray(table['timestamp'].to_pylist()), np.arange(length)/60, atol=1e-5)
            dest = OUTPUT / relative(info, 'data_path', new)
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dest)
            stat = copy.deepcopy(stats_by_id[old])
            stat['episode_index'] = new
            for name in list(stat['stats']):
                if 'image' in name:
                    del stat['stats'][name]
            for name, values in updates.items():
                if name in stat['stats']:
                    stat['stats'][name] = {'min': [int(values.min())], 'max': [int(values.max())], 'mean': [float(values.mean())], 'std': [float(values.std())], 'count': [length]}
            statistics.append(stat)
            result = dict(row)
            labels = [r['task'] for r in tasks if r['task_index'] in set(n_tasks.tolist())]
            result.update(episode_index=new, tasks=labels, short_horizon_task=labels, task_index=int(n_tasks[0]), source_dataset=source.name, source_episode_index=old)
            episodes.append(result)
            mapping.append({'episode_index': new, 'source_dataset': str(source), 'source_episode_index': old, 'length': length})
            for key in KEYS:
                src = source / relative(src_info, 'video_path', old, key)
                assert src.is_file()
                videos.append((src, OUTPUT / relative(info, 'video_path', new, key), length))
            offset += length
            source_frames += length
        assert source_frames == src_info['total_frames']
        source_summary.append({'path': str(source), 'episodes': len(source_rows), 'frames': source_frames, 'hours': source_frames / 216000})
        print('Parquet merged:', source.name, len(episodes), flush=True)
    write_json(meta / 'merge_plan.json', {'sources': source_summary, 'episodes': len(episodes), 'frames': offset, 'hours': offset / 216000, 'room2204_prompt': assembly['task'], 'video_settings': {'height': 224, 'width': 224, 'fps': 60, 'codec': 'libx264', 'pixel_format': 'yuv420p', 'crf': 18, 'preset': 'fast', 'gop': 12, 'bframes': 0, 'scenecut': 0}})
    with ProcessPoolExecutor(max_workers=8) as pool:
        for number, result in enumerate(pool.map(video, videos, chunksize=1), 1):
            if number % 100 == 0 or number == len(videos):
                print(f'Videos verified: {number}/{len(videos)}', flush=True)
    info.update(total_episodes=len(episodes), total_frames=offset, total_videos=len(videos), total_chunks=(len(episodes)+info['chunks_size']-1)//info['chunks_size'], splits={'train': f'0:{len(episodes)}'})
    jsonl(meta / 'episodes.jsonl', episodes)
    jsonl(meta / 'episodes_stats.jsonl', statistics)
    jsonl(meta / 'tasks.jsonl', tasks)
    write_json(meta / 'episode_mapping.json', mapping)
    marker = json.loads((REFERENCE / 'meta/gripper_transform.json').read_text())
    marker.pop('source_dataset', None)
    marker.pop('sides', None)
    marker.update(source_datasets=[str(source) for source in SOURCES], frames=offset,
                  baseline_reapplied=False, note='Inherited already-preprocessed values; no additional gripper transform')
    write_json(meta / 'gripper_transform.json', marker)
    write_json(meta / 'build_provenance.json', {'sources': source_summary, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'curation': 'Inputs already cleaned; no additional cuts or gripper transforms', 'room2204_prompt': assembly['task'], 'image_resize': 'openpi_client.image_tools.resize_with_pad', 'reference_video_copy': True})
    shutil.copyfile(__file__, meta / Path(__file__).name)
    write_json(meta / 'info.json', info)
    write_json(meta / 'pipeline_complete.json', {'state': 'complete', 'all_video_gop_and_timestamps_verified': True, 'episodes': len(episodes), 'frames': offset, 'hours': offset / 216000, 'sources': source_summary})
    print('COMPLETE', OUTPUT, 'hours', offset / 216000, flush=True)


if __name__ == '__main__':
    main()
