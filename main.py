from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from pydub import AudioSegment
import yt_dlp
import time
import os
import logging

logging.basicConfig(level=logging.DEBUG)

def read_urls_from_file(filename):
    with open(filename, 'r') as file:
        urls = file.readlines()
    # Her satırdaki boşlukları ve yeni satır karakterlerini temizle
    urls = [url.strip() for url in urls]
    return urls

# YouTube'dan ses indirme ve dönüştürme işlemini gerçekleştirme
def download_and_convert_to_mp3(url, filename, bitrate='64k'):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'datatobeprocessed/{filename}.%(ext)s',  
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': bitrate.replace('k', ''), 
            }],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            audio_path = f'datatobeprocessed/{filename}.mp3'  # MP3 dosya yolu
        
        print("İndirme ve dönüştürme başarılı! MP3 dosyası:", audio_path)

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

def cut_audio_file(filename, start_time, end_time):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "datatobeprocessed")

    input_path = os.path.join(data_dir, f"{filename}.mp3")
    output_path = os.path.join(data_dir, f"{filename}.mp3")

    try:
        audio = AudioSegment.from_mp3(input_path)

        start_ms = start_time * 60 * 1000  # dakika -> milisaniye
        end_ms = end_time * 60 * 1000  # dakika -> milisaniye
        audio_segment = audio[start_ms:end_ms]

        audio_segment.export(output_path, format="mp3")
        
        print(f"Kesme işlemi başarılı! {start_time} - {end_time} dakika arasındaki kısım {output_path} dosyasına kaydedildi.")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

def split_audio_file(filename, output_dir, start_index, segment_length=6000, min_length=5000):
    try:
        audio = AudioSegment.from_mp3(filename)

        total_length = len(audio)

        current_index = start_index

        for i in range(0, total_length, segment_length):
            segment = audio[i:i+segment_length]
            if len(segment) >= min_length:
                segment_filename = f"{current_index}.mp3"
                segment.export(os.path.join(output_dir, segment_filename), format="mp3")
                current_index += 1
        
        print(f"{filename} dosyası başarıyla {segment_length//1000} saniyelik parçalara ayrıldı.")
        return current_index
    
    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        return start_index

def main():
    # Selenium sürücüsünü başlat
    service = Service(executable_path="chromedriver.exe")
    driver = webdriver.Chrome(service=service)
    driver.maximize_window()

    driver.get('http://youtube.com') 
    searchBox = driver.find_element(By.XPATH, '/html/body/ytd-app/div[1]/div/ytd-masthead/div[4]/div[2]/ytd-searchbox/form/div[1]/div[1]/input')  # Arama kutusunun seçilmesi
    searchBox.send_keys('EGK SOUND')
    time.sleep(10)
    searchButton = driver.find_element(By.XPATH, '/html/body/ytd-app/div[1]/div/ytd-masthead/div[4]/div[2]/ytd-searchbox/button')  # Arama butonun seçilmesi
    searchButton.click()
    current_url = driver.current_url
    print("Mevcut sayfanın URL'si:", current_url)

    filename = "list.txt"
    urls = read_urls_from_file(filename) 
    print("İndirilecek URL'ler:")
    for index, url in enumerate(urls, start=146): 
        print(url)
        download_and_convert_to_mp3(url, filename=str(index), bitrate='64k')

    time.sleep(10)

    for i in range(97, 151, 2):
        cut_audio_file(i, start_time=2, end_time=5)
    

    input_dir = "datatobeprocessed"  
    output_dir = "data"  


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_index = 1

    filenames = sorted(os.listdir(input_dir), key=lambda x: int(os.path.splitext(x)[0]))
    for filename in filenames:
        if filename.endswith(".mp3"):
            file_path = os.path.join(input_dir, filename)
            current_index = split_audio_file(file_path, output_dir, current_index, segment_length=6000)

    time.sleep(10000)
    driver.quit()

if __name__ == "__main__":
    main()
