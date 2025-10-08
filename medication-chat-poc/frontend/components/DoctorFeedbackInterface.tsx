'use client';

import { useState, useEffect } from 'react';
import {
  Brain,
  CheckCircle,
  RefreshCw,
  BarChart3,
  AlertTriangle,
  User,
  TrendingUp,
  Edit3,
  Save,
  X
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';

interface DiagnosisFeedback {
  id: string;
  timestamp: string;
  patientSymptoms: string;
  aiDiagnosis: {
    icdCode: string;
    englishName: string;
    thaiName: string;
    confidence: number;
    medications: Array<{
      englishName: string;
      thaiName: string;
      dosage: string;
    }>;
  };
  doctorFeedback?: {
    approved: boolean;
    correctedDiagnosis?: {
      icdCode: string;
      englishName: string;
      thaiName: string;
    };
    correctedMedications?: Array<{
      englishName: string;
      thaiName: string;
      dosage: string;
    }>;
    notes?: string;
    doctorId: string;
  };
}

interface FeedbackStats {
  total: number;
  approved: number;
  corrected: number;
  mostCommonCorrections: Array<{
    from: string;
    to: string;
    count: number;
  }>;
}

export default function DoctorFeedbackInterface() {
  const [activeTab, setActiveTab] = useState('pending');
  const [pendingFeedback, setPendingFeedback] = useState<DiagnosisFeedback[]>([]);
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [isRetraining, setIsRetraining] = useState(false);
  const [selectedItem, setSelectedItem] = useState<DiagnosisFeedback | null>(null);
  const [isEditing, setIsEditing] = useState(false);

  // Form states for corrections
  const [correctedIcd, setCorrectedIcd] = useState('');
  const [correctedEnglish, setCorrectedEnglish] = useState('');
  const [correctedThai, setCorrectedThai] = useState('');
  const [correctedMedications, setCorrectedMedications] = useState<{
    englishName: string;
    thaiName: string;
    dosage: string;
  }[]>([]);
  const [feedbackNotes, setFeedbackNotes] = useState('');

  useEffect(() => {
    fetchPendingFeedback();
    fetchStats();
  }, []);

  const fetchPendingFeedback = async () => {
    try {
      const response = await fetch('/api/feedback?action=pending');
      const data = await response.json();
      setPendingFeedback(data.feedback || []);
    } catch (error) {
      console.error('Failed to fetch pending feedback:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/feedback?action=stats');
      const data = await response.json();
      setStats(data.stats);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  const submitFeedback = async (
    item: DiagnosisFeedback,
    approved: boolean,
    corrections?: {
      diagnosis?: string;
      medications?: {
        englishName: string;
        thaiName: string;
        dosage: string;
      }[];
    }
  ) => {
    try {
      const feedbackData = {
        action: 'submit_feedback',
        chatId: item.id,
        patientSymptoms: item.patientSymptoms,
        aiDiagnosis: item.aiDiagnosis,
        doctorFeedback: {
          approved,
          doctorId: 'doctor_1', // In real app, get from authentication
          notes: feedbackNotes,
          ...(corrections && {
            correctedDiagnosis: corrections.diagnosis,
            correctedMedications: corrections.medications
          })
        }
      };

      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedbackData)
      });

      if (response.ok) {
        // Remove from pending list
        setPendingFeedback(prev => prev.filter(p => p.id !== item.id));
        setSelectedItem(null);
        setIsEditing(false);
        resetForm();
        fetchStats();
      }
    } catch (error) {
      console.error('Failed to submit feedback:', error);
    }
  };

  const retrainModel = async () => {
    setIsRetraining(true);
    try {
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'retrain_model' })
      });

      if (response.ok) {
        alert('Model retrained successfully!');
        fetchStats();
      }
    } catch (error) {
      console.error('Failed to retrain model:', error);
    } finally {
      setIsRetraining(false);
    }
  };

  const selectItem = (item: DiagnosisFeedback) => {
    setSelectedItem(item);
    setCorrectedIcd(item.aiDiagnosis.icdCode);
    setCorrectedEnglish(item.aiDiagnosis.englishName);
    setCorrectedThai(item.aiDiagnosis.thaiName);
    setCorrectedMedications(item.aiDiagnosis.medications || []);
    setFeedbackNotes('');
  };

  const resetForm = () => {
    setCorrectedIcd('');
    setCorrectedEnglish('');
    setCorrectedThai('');
    setCorrectedMedications([]);
    setFeedbackNotes('');
  };

  // Removed unused functions - addMedication, updateMedication, removeMedication
  // These can be re-added when the medication editing feature is implemented

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-64 bg-white border-r border-gray-200">
        <div className="p-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <Brain className="h-5 w-5 text-blue-600" />
            ศูนย์ฝึกอบรม AI
          </h2>
        </div>

        <nav className="p-4 space-y-2">
          <Button
            variant={activeTab === 'pending' ? 'default' : 'ghost'}
            className="w-full justify-start"
            onClick={() => setActiveTab('pending')}
          >
            <AlertTriangle className="h-4 w-4 mr-2" />
            Pending Reviews ({pendingFeedback.length})
          </Button>

          <Button
            variant={activeTab === 'stats' ? 'default' : 'ghost'}
            className="w-full justify-start"
            onClick={() => setActiveTab('stats')}
          >
            <BarChart3 className="h-4 w-4 mr-2" />
            Training Statistics
          </Button>

          <Button
            variant={activeTab === 'retrain' ? 'default' : 'ghost'}
            className="w-full justify-start"
            onClick={() => setActiveTab('retrain')}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Model Training
          </Button>
        </nav>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Content Area */}
        <div className="flex-1 p-6">
          {activeTab === 'pending' && (
            <div>
              <div className="mb-6">
                <h1 className="text-2xl font-bold text-gray-900 mb-2">
                  Pending Diagnosis Reviews
                </h1>
                <p className="text-gray-600">
                  Review AI diagnoses and provide feedback to improve the system
                </p>
              </div>

              <div className="grid gap-4">
                {pendingFeedback.map((item) => (
                  <Card key={item.id} className="cursor-pointer hover:shadow-md transition-shadow"
                        onClick={() => selectItem(item)}>
                    <CardContent className="p-4">
                      <div className="flex justify-between items-start mb-3">
                        <div className="flex items-center gap-2">
                          <User className="h-4 w-4 text-gray-500" />
                          <span className="text-sm text-gray-600">
                            {new Date(item.timestamp).toLocaleDateString()}
                          </span>
                        </div>
                        <Badge variant="outline" className="text-blue-600 border-blue-600">
                          {item.aiDiagnosis.confidence}% confidence
                        </Badge>
                      </div>

                      <div className="mb-3">
                        <h3 className="font-medium text-gray-900 mb-1">Symptoms:</h3>
                        <p className="text-sm text-gray-700 line-clamp-2">
                          {item.patientSymptoms}
                        </p>
                      </div>

                      <div className="bg-blue-50 p-3 rounded-lg">
                        <h4 className="font-medium text-blue-900 mb-1">AI Diagnosis:</h4>
                        <p className="text-sm text-blue-800">
                          {item.aiDiagnosis.icdCode} - {item.aiDiagnosis.thaiName}
                        </p>
                        {item.aiDiagnosis.medications && item.aiDiagnosis.medications.length > 0 && (
                          <p className="text-xs text-blue-700 mt-1">
                            Medications: {item.aiDiagnosis.medications.map(m => m.englishName).join(', ')}
                          </p>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'stats' && stats && (
            <div>
              <div className="mb-6">
                <h1 className="text-2xl font-bold text-gray-900 mb-2">
                  Training Statistics
                </h1>
                <p className="text-gray-600">
                  View AI performance metrics and training progress
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-blue-100 rounded-lg">
                        <CheckCircle className="h-6 w-6 text-blue-600" />
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Total Reviews</p>
                        <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-green-100 rounded-lg">
                        <TrendingUp className="h-6 w-6 text-green-600" />
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Approval Rate</p>
                        <p className="text-2xl font-bold text-gray-900">
                          {stats.total > 0 ? Math.round((stats.approved / stats.total) * 100) : 0}%
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-orange-100 rounded-lg">
                        <Edit3 className="h-6 w-6 text-orange-600" />
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Corrections</p>
                        <p className="text-2xl font-bold text-gray-900">{stats.corrected}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {stats.mostCommonCorrections.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Most Common Corrections</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {stats.mostCommonCorrections.map((correction, index) => (
                        <div key={index} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                          <div>
                            <span className="text-sm text-gray-600">{correction.from}</span>
                            <span className="text-gray-400 mx-2">→</span>
                            <span className="text-sm font-medium text-gray-900">{correction.to}</span>
                          </div>
                          <Badge variant="outline">{correction.count} times</Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {activeTab === 'retrain' && (
            <div>
              <div className="mb-6">
                <h1 className="text-2xl font-bold text-gray-900 mb-2">
                  Model Training
                </h1>
                <p className="text-gray-600">
                  Apply doctor feedback to improve diagnostic accuracy
                </p>
              </div>

              <Card>
                <CardContent className="p-6">
                  <div className="text-center">
                    <Brain className="h-16 w-16 text-blue-600 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      Ready to Retrain AI Model
                    </h3>
                    <p className="text-gray-600 mb-6">
                      Incorporate all doctor feedback to improve diagnostic patterns and accuracy
                    </p>

                    <Button
                      onClick={retrainModel}
                      disabled={isRetraining}
                      className="px-8"
                    >
                      {isRetraining ? (
                        <>
                          <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                          Retraining Model...
                        </>
                      ) : (
                        <>
                          <Brain className="h-4 w-4 mr-2" />
                          Start Retraining
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </div>

        {/* Detail Panel */}
        {selectedItem && (
          <div className="w-96 bg-white border-l border-gray-200 p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Review Diagnosis</h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedItem(null)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-gray-900 mb-2">Patient Symptoms:</h4>
                <p className="text-sm text-gray-700 bg-gray-50 p-3 rounded-lg">
                  {selectedItem.patientSymptoms}
                </p>
              </div>

              <div>
                <h4 className="font-medium text-gray-900 mb-2">AI Diagnosis:</h4>
                <div className="bg-blue-50 p-3 rounded-lg">
                  <p className="text-sm font-medium text-blue-900">
                    {selectedItem.aiDiagnosis.icdCode} - {selectedItem.aiDiagnosis.englishName}
                  </p>
                  <p className="text-sm text-blue-800">{selectedItem.aiDiagnosis.thaiName}</p>
                  <p className="text-xs text-blue-700 mt-1">
                    Confidence: {selectedItem.aiDiagnosis.confidence}%
                  </p>
                </div>
              </div>

              {!isEditing ? (
                <div className="space-y-3">
                  <Button
                    onClick={() => submitFeedback(selectedItem, true)}
                    className="w-full bg-green-600 hover:bg-green-700"
                  >
                    <CheckCircle className="h-4 w-4 mr-2" />
                    Approve Diagnosis
                  </Button>

                  <Button
                    onClick={() => setIsEditing(true)}
                    variant="outline"
                    className="w-full"
                  >
                    <Edit3 className="h-4 w-4 mr-2" />
                    Make Corrections
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium text-gray-700">ICD Code:</label>
                    <Input
                      value={correctedIcd}
                      onChange={(e) => setCorrectedIcd(e.target.value)}
                      placeholder="e.g., R51"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium text-gray-700">English Name:</label>
                    <Input
                      value={correctedEnglish}
                      onChange={(e) => setCorrectedEnglish(e.target.value)}
                      placeholder="e.g., Headache"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium text-gray-700">Thai Name:</label>
                    <Input
                      value={correctedThai}
                      onChange={(e) => setCorrectedThai(e.target.value)}
                      placeholder="e.g., ปวดศีรษะ"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium text-gray-700">Notes:</label>
                    <Textarea
                      value={feedbackNotes}
                      onChange={(e) => setFeedbackNotes(e.target.value)}
                      placeholder="Additional notes about this correction..."
                      rows={3}
                    />
                  </div>

                  <div className="space-y-2">
                    <Button
                      onClick={() => {
                        const corrections = {
                          diagnosis: `${correctedIcd} - ${correctedEnglish} (${correctedThai})`,
                          medications: correctedMedications
                        };
                        submitFeedback(selectedItem, false, corrections);
                      }}
                      className="w-full"
                    >
                      <Save className="h-4 w-4 mr-2" />
                      Submit Corrections
                    </Button>

                    <Button
                      onClick={() => setIsEditing(false)}
                      variant="outline"
                      className="w-full"
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}