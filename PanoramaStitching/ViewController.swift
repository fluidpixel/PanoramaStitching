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
    
    
    let basePath = "\(NSHomeDirectory())/Desktop/Panorama"
    var inputImages:[String] = []
    var resultPath = ""
    
    override func viewDidLoad() {
        
        super.viewDidLoad()
        
        for index in 0..<18 {
            inputImages.append("\(basePath)/rawImage_\(index).jpg")
        }
        resultPath = "\(basePath)/result.jpg"
        
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
                StitchingInterface.addImagePath(self.inputImages[index], atIndex: index, isLastImage: (index == 17) )
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

