#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "test_data.h"

/**
 * Movement Detect Firmware - TinyML Inference
 * Support pour modèle INT8 quantifié.
 * Adapté pour Raspberry Pi Pico W (RP2040)
 */

// Mettre à 1 pour tester avec un échantillon fixe (sample #42 du test set)
#define TEST_MODE 1 

// --- Configuration Normalisation (Issues de models/normalization.json) ---
const float MU[] = {-0.000636f, -0.000292f, -0.000275f, 0.804749f, 0.028755f, 0.086498f, 0.000506f, -0.000824f, 0.000113f};
const float SIGMA[] = {0.194846f, 0.122427f, 0.106879f, 0.414112f, 0.390995f, 0.357769f, 0.406815f, 0.381854f, 0.255743f};

// --- Paramètres de Quantification (INT8) ---
const float IN_SCALE = 0.061618f;
const int IN_ZERO_POINT = 13;
const float OUT_SCALE = 0.003906f;
const int OUT_ZERO_POINT = -128;

const char* LABELS[] = {"WALKING", "WALK_UP", "WALK_DOWN", "SITTING", "STANDING", "LAYING"};

static tflite::MicroMutableOpResolver<11> resolver;
static tflite::MicroInterpreter* interpreter;
static uint8_t tensor_arena[128 * 1024]; // Arena TFLM pour tensors/intermediaires.

// Historique pour lissage (moyenne glissante / vote majoritaire)
int8_t pred_history[5] = {-1, -1, -1, -1, -1};
int history_idx = 0;
int predictions_count = 0;

void wait_for_serial(unsigned long timeout_ms) {
  unsigned long start = millis();
  while (!Serial && (millis() - start < timeout_ms)) {
    delay(10);
  }
}

void fatal_halt(const char* message) {
  while (true) {
    Serial.println(message);
    delay(1000);
  }
}

void setup() {
  Serial.begin(115200);
  wait_for_serial(8000);
  Serial.println("\n=== Movement Detect Test ===");

  // Enregistrement des opérations nécessaires
  resolver.AddConv2D();
  resolver.AddRelu();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddExpandDims();
  resolver.AddDepthwiseConv2D();
  resolver.AddMean();

  // Chargement du modèle
  auto* model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    fatal_halt("Error: TFLite Schema version mismatch!");
  }

  // Initialisation de l'interpréteur
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, sizeof(tensor_arena));
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    fatal_halt("Error: AllocateTensors failed!");
  }
  
  Serial.println("Model loaded and ready for inference.");
}

void loop() {
  float raw[128][9];
  
  #if TEST_MODE
    // Utilisation des données de test hardcodées (test_data.h)
    memcpy(raw, TEST_SAMPLE, sizeof(raw));
    Serial.print("[TEST] Inference... ");
  #else
    // Lecture réelle du capteur (à implémenter)
    get_imu_window(raw);
  #endif

  // 1️⃣ Quantification de l'entrée : float -> int8
  int8_t* input = interpreter->input(0)->data.int8;
  for(int i=0; i<128; i++) {
    for(int c=0; c<9; c++) {
      //float normalized = (raw[i][c] - MU[c]) / SIGMA[c];
      // Formule : quantized = normalized / scale + zero_point
      int32_t q = (int32_t)(raw[i][c] / IN_SCALE + IN_ZERO_POINT);
      // Saturation pour rester dans [-128, 127]
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      input[i * 9 + c] = (int8_t)q;
    }
  }

  // 2️⃣ Invocation du modèle
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Error: Invoke failed!");
    return;
  }

  // 3️⃣ Déquantification de la sortie : int8 -> probabilities (float)
  int8_t* output = interpreter->output(0)->data.int8;
  int current_pred = 0;
  float max_prob = -1.0f;
  
  for(int i=0; i<6; i++) {
    // Formule : float = (quantized - zero_point) * scale
    float prob = (float)(output[i] - OUT_ZERO_POINT) * OUT_SCALE;
    if(prob > max_prob) {
      max_prob = prob;
      current_pred = i;
    }
  }

  // 4️⃣ Lissage par vote majoritaire sur les 5 dernières prédictions
  pred_history[history_idx] = current_pred;
  history_idx = (history_idx + 1) % 5;
  
  if (predictions_count < 5) {
    predictions_count++;
  }

  // 5️⃣ Affichage des résultats (seulement après 3 prédictions)
  if (predictions_count >= 3) {
    int counts[6] = {0};
    for(int i=0; i<5; i++) {
      if (pred_history[i] >= 0) {
        counts[pred_history[i]]++;
      }
    }
    
    int final_pred = 0;
    for(int i=1; i<6; i++) {
      if(counts[i] > counts[final_pred]) {
        final_pred = i;
      }
    }
    
    Serial.print("Result: ");
    Serial.print(LABELS[final_pred]);
    Serial.print(" (Confidence: ");
    Serial.print(max_prob * 100, 1);
    Serial.println("%)");
  } else {
    Serial.print("Warming up... (");
    Serial.print(predictions_count);
    Serial.println("/3)");
  }
  
  delay(2000);
}

// Placeholder pour la lecture IMU
void get_imu_window(float buf[][9]) {
  // Implémenter ici la lecture I2C (ex: MPU6050) pour remplir la fenêtre de 128 samples
  for(int i=0; i<128; i++) {
    for(int c=0; c<9; c++) {
      buf[i][c] = 0.0f; 
    }
  }
}
