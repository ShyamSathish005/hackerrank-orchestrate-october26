import Foundation
import Vision
import AppKit

let fileManager = FileManager.default
let arguments = CommandLine.arguments

guard arguments.count > 1 else {
    print("Usage: swift ocr.swift <image_path>")
    exit(1)
}

let imagePath = arguments[1]
let imageURL = URL(fileURLWithPath: imagePath)

guard let image = NSImage(contentsOf: imageURL),
      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    exit(1)
}

let request = VNRecognizeTextRequest { (request, error) in
    guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
    for observation in observations {
        if let topCandidate = observation.topCandidates(1).first {
            print(topCandidate.string)
        }
    }
}

request.recognitionLevel = .accurate
let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
try? handler.perform([request])
