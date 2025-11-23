import streamlit as st
import os
import time
from pathlib import Path
import wave
import pyaudio
from pydub import AudioSegment
from pydub.effects import normalize
from audiorecorder import audiorecorder
import numpy as np
from scipy.io.wavfile import write
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
import constants as ct

def preprocess_audio(audio_file_path):
    """
    音声ファイルの前処理（ノイズ除去と音量正規化）
    Args:
        audio_file_path: 音声ファイルのパス
    Returns:
        前処理済み音声ファイルのパス
    """
    try:
        # 音声ファイルを読み込み
        audio = AudioSegment.from_wav(audio_file_path)
        
        # 音量の正規化（音声認識精度向上のため）
        normalized_audio = normalize(audio)
        
        # 無音部分のトリミング（前後の余計な無音を削除）
        # silence_thresh: 無音とみなす閾値（dB）、min_silence_len: 無音の最小長（ms）
        trimmed_audio = normalized_audio.strip_silence(
            silence_thresh=-40,  # -40dB以下を無音とする
            silence_len=300,     # 300ms以上の無音を検出
            padding=200          # トリミング後に前後200msの余白を残す
        )
        
        # 前処理済みファイルとして保存
        preprocessed_path = audio_file_path.replace(".wav", "_preprocessed.wav")
        trimmed_audio.export(preprocessed_path, format="wav", parameters=["-ar", "16000"])
        
        return preprocessed_path
    except Exception as e:
        # エラーが発生した場合は元のファイルをそのまま返す
        print(f"音声前処理エラー: {e}")
        return audio_file_path

def record_audio(audio_input_file_path):
    """
    音声入力を受け取って音声ファイルを作成
    """

    audio = audiorecorder(
        start_prompt="発話開始",
        pause_prompt="やり直す",
        stop_prompt="発話終了",
        start_style={"color":"white", "background-color":"black"},
        pause_style={"color":"gray", "background-color":"white"},
        stop_style={"color":"white", "background-color":"black"}
    )

    if len(audio) > 0:
        # 音声認識精度向上のため、16kHzでエクスポート
        audio.export(audio_input_file_path, format="wav", parameters=["-ar", "16000"])
    else:
        st.stop()

def transcribe_audio(audio_input_file_path, context_prompt=None):
    """
    音声入力ファイルから文字起こしテキストを取得
    Args:
        audio_input_file_path: 音声入力ファイルのパス
        context_prompt: 文脈を提供するプロンプト（オプション）
    """

    # 音声ファイルの前処理（ノイズ除去と音量正規化）
    preprocessed_path = preprocess_audio(audio_input_file_path)
    
    with open(preprocessed_path, 'rb') as audio_input_file:
        # Whisper APIのパラメータ最適化
        params = {
            "model": "whisper-1",
            "file": audio_input_file,
            "language": "en",
            "temperature": 0.0  # 確実性を高めるため温度を0に設定
        }
        
        # 文脈プロンプトがある場合は追加（精度向上のため）
        if context_prompt:
            params["prompt"] = context_prompt
        
        transcript = st.session_state.openai_obj.audio.transcriptions.create(**params)
    
    # 前処理済みファイルを削除
    if preprocessed_path != audio_input_file_path:
        os.remove(preprocessed_path)
    
    # 音声入力ファイルを削除
    os.remove(audio_input_file_path)

    return transcript

def save_to_wav(llm_response_audio, audio_output_file_path):
    """
    一旦mp3形式で音声ファイル作成後、wav形式に変換
    Args:
        llm_response_audio: LLMからの回答の音声データ
        audio_output_file_path: 出力先のファイルパス
    """

    temp_audio_output_filename = f"{ct.AUDIO_OUTPUT_DIR}/temp_audio_output_{int(time.time())}.mp3"
    with open(temp_audio_output_filename, "wb") as temp_audio_output_file:
        temp_audio_output_file.write(llm_response_audio)
    
    audio_mp3 = AudioSegment.from_file(temp_audio_output_filename, format="mp3")
    audio_mp3.export(audio_output_file_path, format="wav")

    # 音声出力用に一時的に作ったmp3ファイルを削除
    os.remove(temp_audio_output_filename)

def play_wav(audio_output_file_path, speed=1.0):
    """
    音声ファイルの読み上げ
    Args:
        audio_output_file_path: 音声ファイルのパス
        speed: 再生速度（1.0が通常速度、0.5で半分の速さ、2.0で倍速など）
    """

    # 音声ファイルの読み込み
    audio = AudioSegment.from_wav(audio_output_file_path)
    
    # 速度を変更
    if speed != 1.0:
        # frame_rateを変更することで速度を調整
        modified_audio = audio._spawn(
            audio.raw_data, 
            overrides={"frame_rate": int(audio.frame_rate * speed)}
        )
        # 元のframe_rateに戻すことで正常再生させる（ピッチを保持したまま速度だけ変更）
        modified_audio = modified_audio.set_frame_rate(audio.frame_rate)

        modified_audio.export(audio_output_file_path, format="wav")

    # PyAudioで再生
    with wave.open(audio_output_file_path, 'rb') as play_target_file:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(play_target_file.getsampwidth()),
            channels=play_target_file.getnchannels(),
            rate=play_target_file.getframerate(),
            output=True
        )

        data = play_target_file.readframes(1024)
        while data:
            stream.write(data)
            data = play_target_file.readframes(1024)

        stream.stop_stream()
        stream.close()
        p.terminate()
    
    # LLMからの回答の音声ファイルを削除
    os.remove(audio_output_file_path)

def create_chain(system_template):
    """
    LLMによる回答生成用のChain作成
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    chain = ConversationChain(
        llm=st.session_state.llm,
        memory=st.session_state.memory,
        prompt=prompt
    )

    return chain

def create_problem_and_play_audio():
    """
    問題生成と音声ファイルの再生
    Args:
        chain: 問題文生成用のChain
        speed: 再生速度（1.0が通常速度、0.5で半分の速さ、2.0で倍速など）
        openai_obj: OpenAIのオブジェクト
    """

    # 問題文を生成するChainを実行し、問題文を取得
    problem = st.session_state.chain_create_problem.predict(input="")

    # LLMからの回答を音声データに変換
    llm_response_audio = st.session_state.openai_obj.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=problem
    )

    # 音声ファイルの作成
    audio_output_file_path = f"{ct.AUDIO_OUTPUT_DIR}/audio_output_{int(time.time())}.wav"
    save_to_wav(llm_response_audio.content, audio_output_file_path)

    # 音声ファイルの読み上げ
    play_wav(audio_output_file_path, st.session_state.speed)

    return problem, llm_response_audio

def create_evaluation():
    """
    ユーザー入力値の評価生成
    """

    llm_response_evaluation = st.session_state.chain_evaluation.predict(input="")

    return llm_response_evaluation

def check_grammar(user_response):
    """
    ユーザーの英語回答を分析し、文法・表現の誤りを指摘して改善提案を生成
    Args:
        user_response: ユーザーの英語回答テキスト
    Returns:
        文法チェック結果のテキスト
    """
    system_template = ct.SYSTEM_TEMPLATE_GRAMMAR_CHECK.format(user_response=user_response)
    chain_grammar_check = create_chain(system_template)
    grammar_feedback = chain_grammar_check.predict(input="")
    
    return grammar_feedback

def check_grammar_summary(user_responses):
    """
    会話終了時に複数の発話をまとめて分析し、文法・表現の誤りを指摘して改善提案を生成
    Args:
        user_responses: ユーザーの英語発話のリスト
    Returns:
        文法チェック結果のテキスト
    """
    # 発話を番号付きで整形
    responses_text = ""
    for i, response in enumerate(user_responses, 1):
        responses_text += f"{i}. \"{response}\"\n"
    
    system_template = ct.SYSTEM_TEMPLATE_GRAMMAR_CHECK_SUMMARY.format(user_responses=responses_text)
    chain_grammar_check_summary = create_chain(system_template)
    grammar_feedback_summary = chain_grammar_check_summary.predict(input="")
    
    return grammar_feedback_summary