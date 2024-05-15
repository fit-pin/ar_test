package com.seok;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.UUID;

import org.json.JSONException;
import org.json.JSONObject;

public class App {

    public JSONObject fileupload(String url, String fileName, float personKey) throws IOException, FileNotFoundException, JSONException {
        URL link = new URL(url);
        HttpURLConnection huc = (HttpURLConnection) link.openConnection();
        // http 연결 부분

        String boundary = UUID.randomUUID().toString(); // 요청을 구분하기 위한 코드
        huc.setRequestMethod("POST");
        huc.setRequestProperty("Connection", "Keep-Alive");
        huc.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
        huc.setDoOutput(true);
        // 파일 전송을 위한 해더 설정

        DataOutputStream dos = new DataOutputStream(huc.getOutputStream()); // 해더작성을 하기위한 객체

        dos.writeBytes("--" + boundary + "\r\n");
        dos.writeBytes("Content-Disposition: form-data; name=\"anaFile\"; filename=\"" + fileName + "\"" + "\r\n");
        dos.writeBytes("\r\n");
        // 파일이 전송되는 부분

        FileInputStream fis = new FileInputStream(fileName);
        byte[] buffer = new byte[1024]; // 버퍼 크기설정
        int bytesRead;
        while ((bytesRead = fis.read(buffer)) != -1) {
            dos.write(buffer, 0, bytesRead);
        }
        fis.close();
        // 파일 내용 전송

        dos.writeBytes("\r\n");
        dos.writeBytes("--" + boundary + "\r\n");
        dos.writeBytes("Content-Disposition: form-data; name=\"personKey\"" + "\r\n");
        dos.writeBytes("\r\n");
        dos.write(String.valueOf(personKey).getBytes());

        dos.writeBytes("\r\n");
        dos.writeBytes("--" + boundary + "--" + "\r\n");
        dos.flush();
        dos.close();
        // 마지막 바운더리 추가

        if (huc.getResponseCode() == HttpURLConnection.HTTP_OK) { // 맞는 응답인지 확인
            // 정상응답
            BufferedReader br = new BufferedReader(new InputStreamReader(huc.getInputStream(), "utf-8"));
            String json = br.readLine();
            return new JSONObject(json); // 정상 응답일 경우 리턴
        } else {
            // 비정상 응답
            BufferedReader br = new BufferedReader(new InputStreamReader(huc.getInputStream(), "utf-8"));
            String json = br.readLine();
            return  new JSONObject(json);
        }
    }
    public static void main(String[] args) throws Exception {
        App app = new App();
        JSONObject result = app.fileupload("http://korseok.kro.kr/bodymea", "test.jpg", 174);
        System.err.println(result);

    }
}
