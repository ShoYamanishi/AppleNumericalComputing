//
//  ContentView.swift
//  iOSTester_04
//
//  Created by Shoichiro Yamanishi on 19.04.22.
//

import SwiftUI

struct ContentView: View {

    @State private var running = false
    
    var body: some View {
        VStack(alignment: .leading) {
            Button("Run") {
                if !running {
                    running = true
                    let runner = TestPrefixSumObjc()
                    runner!.run()
                    print("finished!")
                }
            }.font(.system(size: 60))
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}