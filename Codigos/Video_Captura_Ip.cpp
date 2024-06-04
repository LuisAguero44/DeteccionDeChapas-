#include <iostream>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

using namespace cv;
using namespace std;

// Función para obtener los nombres de las capas de salida de YOLO
vector<String> getOutputsNames(const dnn::Net& net) {
    static vector<String> names;
    if (names.empty()) {
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector<String> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

int main() {
    // Inicializar Tesseract OCR
    tesseract::TessBaseAPI api;
    api.Init(NULL, "eng", tesseract::OEM_DEFAULT);

    // Cargar el modelo de detección de objetos YOLO
    dnn::Net net = dnn::readNetFromONNX("/home/coliveri/best.onnx");
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);

    // Inicializamos el detector de movimiento
    Ptr<BackgroundSubtractorMOG2> detection = createBackgroundSubtractorMOG2(10000, 12);

    // Directorio de videos
    string pathDirsVideos = "/media/gpu";
    vector<string> dirVideos;
    glob(pathDirsVideos + "/*", dirVideos);
    // Path de cada video
    string pathVideo;

    // Ruteado a directorios de videos
    for (const auto& dir : dirVideos) {
        vector<string> pathVideos;
        glob(dir + "/*", pathVideos);
        for (const auto& video : pathVideos) {
            // Captura de los fotogramas del video
            VideoCapture capture(video);
            if (!capture.isOpened()) {
                cerr << "Error al abrir el video: " << video << endl;
                continue;
            }

            while (true) {
                Mat frame;
                capture >> frame;
                if (frame.empty())
                    break;

                // Dibujar rectángulo en el área de interés
                rectangle(frame, Rect(350, 250, 1085 - 350, 560 - 250), Scalar(0, 255, 0), 2);
                Mat recorteDet = frame(Range(250, 560), Range(350, 1085));
                Mat recorteMov = frame(Range(200, 250), Range(350, 750));

                // Máscara de detección de movimiento
                Mat mascara;
                detection->apply(recorteMov, mascara);
                threshold(mascara, mascara, 254, 255, THRESH_BINARY);

                vector<vector<Point>> contornos;
                findContours(mascara, contornos, RETR_TREE, CHAIN_APPROX_SIMPLE);
                sort(contornos.begin(), contornos.end(), [](const vector<Point>& a, const vector<Point>& b) {
                    return contourArea(a) > contourArea(b);
                });

                for (const auto& contorno : contornos) {
                    double area = contourArea(contorno);
                    if (area > 5000) {
                        // Detección utilizando YOLO
                        Mat blob = dnn::blobFromImage(recorteDet, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
                        net.setInput(blob);
                        vector<Mat> outs;
                        net.forward(outs, getOutputsNames(net));

                        // Lectura de la chapa utilizando Tesseract OCR
                        Mat placa = recorteDet.clone();
                        Mat placaGray;
                        cvtColor(placa, placaGray, COLOR_BGR2GRAY);
                        api.SetImage(placaGray.data, placaGray.cols, placaGray.rows, 1, placaGray.cols);
                        char* outText = api.GetUTF8Text();
                        string chapa(outText);
                        cout << "Chapa: " << chapa << endl;
                        delete[] outText;

                        imshow("Detector de Chapas", frame);
                    }
                }

                if (waitKey(1) == 27)
                    break;
            }
        }
    }

    destroyAllWindows();
    return 0;
}
