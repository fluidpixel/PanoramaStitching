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
    
    var imageAttitudes:[(w:Double, x:Double, y:Double, z:Double)] = [
        (0.211378,	0.230308,	0.705061,	0.636527),
        (0.35838,	0.369747,	0.641036,	0.569144),
        (0.461144,	0.487013,	0.565393,	0.480099),
        (0.55402,	0.567375,	0.465565,	0.392935),
        (0.617761,	0.627596,	0.364741,	0.302422),
        (0.708924,	0.627455,	0.240336,	0.214398),
        (0.748001,	0.634226,	0.15022,	0.125242),
        (0.765404,	0.637805,	0.0739526,	0.0434995),
        (0.771624,	0.634829,	-0.00571326,	-0.0394398),
        (0.7549,	0.639066,	-0.0831885,	-0.121659),
        (0.741092,	0.617709,	-0.162128,	-0.207202),
        (0.703308,	0.592275,	-0.255544,	-0.298772),
        (0.639939,	0.559324,	-0.361379,	-0.383457),
        (0.532003,	0.507842,	-0.494965,	-0.462686),
        (0.435126,	0.405542,	-0.589413,	-0.546619),
        (0.300774,	0.274011,	-0.676073,	-0.614311),
        (0.149594,	0.115304,	-0.739599,	-0.646003),
        (-0.0119513,	-0.0582277,	-0.74929,	-0.659569)
    ]
    
    override func viewDidLoad() {
        
        super.viewDidLoad()
        
        for index in 0..<18 {
            let imagePath = NSBundle.mainBundle().pathForResource("rawImage_\(index)", ofType: "jpg", inDirectory: "SampleImages")!
            inputImages.append(imagePath)
        }
        
        resultPath = inputImages[0].stringByDeletingLastPathComponent.stringByAppendingPathComponent("result.jpg")
        
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

