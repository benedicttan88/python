# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 17:01:53 2017

@author: WB512563
"""

import QuantLib as ql




def QL_makeVanillaSwap(flavour, Notional,fixedRate, fixedSchedule, floatSchedule, indexCurve, floatSpread ):
    ## Single Currency Swap
    """
    Takes in
    flavour = Payer/Receiver
    Notional
    fixedRate
    QL fixedSchedule
    QL floatSchedule
    QL IborIndex
    floatSpread
    
    """
    fixedLegDC = ql.Actual360()

    swap = ql.VanillaSwap(flavour, 
                      Notional,
                      fixedSchedule,
                      fixedRate,
                      fixedLegDC,
                      floatSchedule,
                      indexCurve,
                      floatSpread,
                      indexCurve.dayCounter())
    
    return swap


def QL_makeCCSwap():
    pass

def QL_makeBasisSwap(flavour, Notional_1, Notional_2, paySchedule, receiveSchedule, payIndex, receiveIndex, 
                     paySpread, receiveSpread, payDayCount, receiveDayCount):
    

    BasisSwap = ql.Swap
    
    
    
    
    
    pass

def QL_makeCCBasisSwap():
    pass





def BasisSwap(payNotional, receiveNotional, paySchedule, receiveSchedule, payIndex, receiveIndex, paySpread, receiveSpread):
    

    pass

    
    
    



class QL_BasisSwap(object):
    
        
    def __init__(self,nowdate, payNotional,receiveNotional, paySchedule, receiveSchedule, payIndex, receiveIndex, paySpread, receiveSpread,
                 payDiscountCurve, payForwardCurve, receiveDiscountCurve, receiveForwardCurve):
        
        self.nowdate = nowdate                  #QuantLib Date
        
        self.payNotional = payNotional
        self.receiveNotional = receiveNotional
        self.paySchedule = paySchedule
        self.receiveSchedule = receiveSchedule
        
        self.paySpread = paySpread
        self.receiveSpread = receiveSpread
    
        self.payDayCount = ql.Actual365Fixed()
        self.receiveDayCount = ql.Actual360()
        
        self.payDscCurve = payDiscountCurve
        self.payFwdCurve = payForwardCurve
        
        self.receiveDscCurve = receiveDiscountCurve
        self.receiveFwdCurve = receiveForwardCurve

    
    
    
    def fairSpread(self):
        
        pay_schedule_dates = [y for (x,y) in enumerate(self.paySchedule)]
        #receive_schedule_dates = [y for (x,y) in enumerate(self.receiveSchedule)]
        
        AnnuityF = 0.0;
        for i in range(1,len(pay_schedule_dates)):
            
            payDF = self.payDscCurve.discount(pay_schedule_dates[i])
            payAccural = pay_schedule_dates[i] - pay_schedule_dates[i] / 360
            AnnuityF += payDF * payAccural                    
                                           
        
        
        
    
    def PV_ReceiveLeg(self):
        
        receive_schedule_dates = [y for (x,y) in enumerate(self.receiveSchedule)]
        
        sum_ = 0.0
        for i in range(1,len(receive_schedule_dates)):
            
            receiveDF = self.receiveDscCurve.discount(receive_schedule_dates[i])
            receiveFwd = self.receiveFwdCurve.forwardRate()
            receiveAccural = receive_schedule_dates[i] - receive_schedule_dates[i-1] / 360
                   
            sum_ += (receiveDF * receiveFwd * receiveAccural)
        
        return sum_
        

    def PV_PayLeg(self):
        
        fairSpread = self.fairSpread()
        
        pay_schedule_dates = [y for (x,y) in enumerate(self.paySchedule)]
        
        sum_ = 0.0;
        for i in range(1,len(pay_schedule_dates)):
            
            payDF = self.payDscCurve.discount(pay_schedule_dates[i])
            payFwd = self.receiveFwdCurve.forwardRate(pay_schedule_dates[i]) - fairSpread
            payAccural = pay_schedule_dates[j] - pay_schedule_dates[j-1] / 360
            
            sum_ += (payDF * payFwd * payAccural)
                                           
        return sum_
    
        
    



