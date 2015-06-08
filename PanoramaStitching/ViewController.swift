//
//  ViewController.swift
//  PanoramaStitching
//
//  Created by Paul on 02/06/2015.
//  Copyright (c) 2015 Fluid Pixel. All rights reserved.
//

import Cocoa

class ViewController: NSViewController, NSTableViewDataSource, NSTableViewDelegate, StitchingInterfaceProtocol {

    @IBOutlet var tableView:NSTableView!
    @IBOutlet var tableViewOrderColumn:NSTableColumn!
    @IBOutlet var tableViewPathColumn:NSTableColumn!
    
    @IBOutlet var resultText:NSTextField!
    @IBOutlet var progressText:NSTextField!
    @IBOutlet var stitchButton:NSButton!
    
    var inputImages:[String] = []
    var resultPath = ""
    
    var imageAttitudes:[(w:Double, x:Double, y:Double, z:Double)] = []
    
    override func viewDidLoad() {
        
        super.viewDidLoad()
        
        for index in 0..<18 {
            let imagePath = NSBundle.mainBundle().pathForResource("rawImage_\(index)", ofType: "jpg", inDirectory: "SampleImages")!
            inputImages.append(imagePath)
        }
        
        resultPath = inputImages[0].stringByDeletingLastPathComponent.stringByAppendingPathComponent("result.jpg")
        
        let attitudePath = NSBundle.mainBundle().pathForResource("raw_attitude", ofType: "txt", inDirectory: "SampleImages")!
        let attitudeFile = String(contentsOfFile: attitudePath, encoding: NSUTF8StringEncoding, error: nil)!
        let attitudeData = attitudeFile.componentsSeparatedByString("\n").map { $0.componentsSeparatedByString(",\t") }.filter { $0.count == 5 }
        imageAttitudes.reserveCapacity(attitudeData.count)
        for i in 0..<attitudeData.count {
            let index = (attitudeData[i][0] as NSString).integerValue
            let w = (attitudeData[i][1] as NSString).doubleValue
            let x = (attitudeData[i][2] as NSString).doubleValue
            let y = (attitudeData[i][3] as NSString).doubleValue
            let z = (attitudeData[i][4] as NSString).doubleValue
            imageAttitudes.insert((w, x, y, z), atIndex: index)
        }
        
        self.tableView.setDataSource(self)
        self.tableView.setDelegate(self)
        resultText.stringValue = resultPath
        
        progressText.stringValue = "Ready"

        // Do any additional setup after loading the view.
    }

    override var representedObject: AnyObject? {
        didSet {
        // Update the view, if already loaded.
        }
    }

    // MARK: Actions
    @IBAction func stitchButtonPressed(sender:NSButton) {
    
        StitchingInterface.setDelegate(self)
        
        self.stitchButton.enabled = false
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0)) {
            StitchingInterface.initaliseAlgorithm(0, withResultPath: self.resultPath, withPreviewPath: "")
            
            for index in 0..<18 {
                StitchingInterface.addImagePath(self.inputImages[index], atIndex: index, isLastImage: (index == 17),
                    withAttitudeX: self.imageAttitudes[index].x,
                                y: self.imageAttitudes[index].y,
                                z: self.imageAttitudes[index].z,
                                w: self.imageAttitudes[index].w)
            }
            
            self.setText("Done")
            self.stitchButton.enabled = true
        }
    }
    // MARK: StitchingInterfaceProtocol
    func setText(text: String!) -> Void {
        dispatch_async(dispatch_get_main_queue()) {
            self.progressText.stringValue = text
        }
    }
    func setProgress(progress: Double) {
        // TODO:
//        dispatch_async(dispatch_get_main_queue()) {
//        }
    }
    func didFinishProcessingSuccessfully(resultPath: String!) {
        dispatch_async(dispatch_get_main_queue()) {
            PanoramaStitchingApp?.openPanoramaViewer(resultPath)
        }
    }
    
    // MARK: NSTableViewDataSource
    func numberOfRowsInTableView(tableView: NSTableView) -> Int {
        if tableView == self.tableView {
            return inputImages.count
        }
        return 0
    }
    func tableView(tableView: NSTableView, objectValueForTableColumn tableColumn: NSTableColumn?, row: Int) -> AnyObject? {
        if tableView == self.tableView {
            if tableColumn == self.tableViewOrderColumn {
                return "\(row)"
            }
            else if tableColumn == self.tableViewPathColumn {
                return inputImages[row]
            }
        }
        return "?" as NSString
    }
    

//  func tableView(tableView: NSTableView, setObjectValue object: AnyObject?, forTableColumn tableColumn: NSTableColumn?, row: Int)

    
}

