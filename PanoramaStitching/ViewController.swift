//
//  ViewController.swift
//  PanoramaStitching
//
//  Created by Paul on 02/06/2015.
//  Copyright (c) 2015 Fluid Pixel. All rights reserved.
//

import Cocoa

class ViewController: NSViewController, NSTableViewDataSource, NSTableViewDelegate {

    @IBOutlet var tableView:NSTableView!
    @IBOutlet var tableViewOrderColumn:NSTableColumn!
    @IBOutlet var tableViewPathColumn:NSTableColumn!
    
    @IBOutlet var resultText:NSTextField!
    
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
        

        // Do any additional setup after loading the view.
    }

    override var representedObject: AnyObject? {
        didSet {
        // Update the view, if already loaded.
        }
    }

    // MARK: Actions
    @IBAction func stitchButtonPressed(sender:NSButton) {
        
        StitchingInterface.initaliseAlgorithm(0, withResultPath: resultPath, withPreviewPath: "")
        
        for index in 0..<18 {
            StitchingInterface.addImagePath(inputImages[index], atIndex: index, isLastImage: (index == 17) )
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

